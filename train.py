import os
import json
import math
import random
import sys
from types import SimpleNamespace
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


"""Training entry point for the four-branch gated-attention fusion model.

The model expects each sample to be stored as one JSON file. Each JSON file
contains radiomics-style image features, Doppler/color-flow features, clinical
metadata, an embryo type field, and a binary label. The code deliberately keeps
the data parser close to the model so that external users can adapt the field
names to their own institutional schema without changing the network itself.
"""

# =========================
# 1. Fixed field definitions
# =========================
# Embryo type is intentionally removed from the general clinical branch and
# encoded as an independent fourth branch. This lets the fusion module learn
# how embryo information interacts with image, Doppler, and clinical tokens.
CLINICAL_KEYS = [
    "age",
    "uterine_position",
    "endometrial_thickness_mm",
    "endometrial_pattern",
    "antegrade_peristalsis",
    "peristalsis_direction",
    "endometrial_volume_ml",
    "endometrial_blood_flow_sd",
    "endometrial_blood_flow_pi",
    "endometrial_blood_flow_ri",
    "vascularization_index",
    "flow_index",
    "vascularization_flow_index",
    "maternal_bmi",
    "infertility_duration_years",
]

# These clinical fields are categorical and are therefore embedded.
CLINICAL_CAT_KEYS = [
    "uterine_position",
    "endometrial_pattern",
    "antegrade_peristalsis",
    "peristalsis_direction",
]

# Embryo type field used by the independent embryo branch.
EMBRYO_TYPE_KEY = "embryo_type"

# Token order used throughout training, inference, and interpretability export.
TOKEN_NAMES = ["image", "doppler", "clinical", "embryo_type"]


# =========================
# 2. Utility functions
# =========================
def set_seed(seed: int = 42):
    """Set random seeds for reproducible training where possible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

def configure_utf8_stdio() -> None:
    # Ensure logs are emitted as UTF-8 when redirected to files.
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def find_json_files(root_dir: str) -> List[str]:
    """Recursively find all JSON files under a directory."""
    root = Path(root_dir)
    return sorted([str(p) for p in root.rglob("*.json")])


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_float(x: Any) -> float:
    """Convert a value to float when possible; return np.nan on failure."""
    if x is None:
        return np.nan
    if isinstance(x, bool):
        return float(x)
    if isinstance(x, (int, float)):
        return float(x)
    x = str(x).strip()
    if x == "" or x.lower() in {"nan", "none", "null"}:
        return np.nan
    try:
        return float(x)
    except Exception:
        return np.nan


def safe_label(x: Any) -> int:
    """Read a binary label that can be converted to 0 or 1."""
    v = safe_float(x)
    if np.isnan(v):
        raise ValueError(f"label is empty or cannot be parsed: {x}")
    return int(round(v))


def is_image_feature_key(k: str) -> bool:
    """
    Radiomics, shape, and texture features are routed to the image branch.
    By convention, keys starting with `original_` are treated as image features.
    """
    return k.startswith("original_")


def is_doppler_feature_key(k: str) -> bool:
    """
    All other keys in `features` are routed to the Doppler branch,
    such as red, blue, color, or has_blue measurements.
    """
    return not is_image_feature_key(k)


# =========================
# 3. Metadata construction
# =========================
def build_feature_spaces(train_jsons: List[str], val_jsons: List[str]) -> Dict[str, Any]:
    """
    Infer the following metadata from train and validation JSON files:
    1) image feature names
    2) Doppler feature names
    3) clinical continuous feature names
    4) clinical categorical feature names
    5) embryo_type category mapping

    Notes:
    - Category mappings are built from the training split only to avoid leakage.
    - Unseen validation categories are mapped to `__UNK__`.
    """
    train_records = [load_json(p) for p in train_jsons]
    val_records = [load_json(p) for p in val_jsons]
    all_records = train_records + val_records

    # Collect all structured feature names.
    all_feature_keys = set()
    for rec in all_records:
        feats = rec.get("features", {}) or {}
        all_feature_keys.update(feats.keys())

    image_keys = sorted([k for k in all_feature_keys if is_image_feature_key(k)])
    doppler_keys = sorted([k for k in all_feature_keys if is_doppler_feature_key(k)])

    # Collect fields that appear in clinical_info.
    clinical_keys_present = set()
    for rec in all_records:
        cli = rec.get("clinical_info", {}) or {}
        clinical_keys_present.update(cli.keys())

    # Use the fixed allowlist and keep only fields present in the data.
    clinical_keys = [k for k in CLINICAL_KEYS if k in clinical_keys_present]
    cat_keys = [k for k in CLINICAL_CAT_KEYS if k in clinical_keys]
    cont_keys = [k for k in clinical_keys if k not in cat_keys]

    # ===== Build clinical category mappings =====
    cat_maps = {}
    for k in cat_keys:
        values = []
        for rec in train_records:
            raw = (rec.get("clinical_info", {}) or {}).get(k, None)
            if raw is None:
                continue
            values.append(str(raw))
        uniq = sorted(set(values))
        mapping = {"__UNK__": 0}
        for idx, val in enumerate(uniq, start=1):
            mapping[val] = idx
        cat_maps[k] = mapping

    # ===== Build embryo_type category mapping =====
    embryo_values = []
    for rec in train_records:
        raw = (rec.get("clinical_info", {}) or {}).get(EMBRYO_TYPE_KEY, None)
        if raw is None:
            continue
        embryo_values.append(str(raw))
    embryo_uniq = sorted(set(embryo_values))
    embryo_type_map = {"__UNK__": 0}
    for idx, val in enumerate(embryo_uniq, start=1):
        embryo_type_map[val] = idx

    return {
        "image_keys": image_keys,
        "doppler_keys": doppler_keys,
        "clinical_cont_keys": cont_keys,
        "clinical_cat_keys": cat_keys,
        "cat_maps": cat_maps,
        "embryo_type_key": EMBRYO_TYPE_KEY,
        "embryo_type_map": embryo_type_map,
    }


def build_array_and_labels(json_paths: List[str], meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a list of JSON paths into numpy arrays.
    Returned fields:
    - image
    - doppler
    - clinical_cont
    - clinical_cat
    - embryo_type
    - labels
    - patient_ids
    - file_paths
    """
    image_keys = meta["image_keys"]
    doppler_keys = meta["doppler_keys"]
    cont_keys = meta["clinical_cont_keys"]
    cat_keys = meta["clinical_cat_keys"]
    cat_maps = meta["cat_maps"]
    embryo_type_map = meta["embryo_type_map"]

    image_arr = []
    doppler_arr = []
    clinical_cont_arr = []
    clinical_cat_arr = []
    embryo_type_arr = []
    labels = []
    patient_ids = []
    file_paths = []

    for path in json_paths:
        rec = load_json(path)
        cli = rec.get("clinical_info", {}) or {}
        feats = rec.get("features", {}) or {}

        # Image and Doppler branches
        img_vec = [safe_float(feats.get(k, np.nan)) for k in image_keys]
        dop_vec = [safe_float(feats.get(k, np.nan)) for k in doppler_keys]

        # Clinical continuous features
        cli_cont_vec = [safe_float(cli.get(k, np.nan)) for k in cont_keys]

        # Clinical categorical features
        cli_cat_vec = []
        for k in cat_keys:
            raw = cli.get(k, None)
            raw_key = str(raw) if raw is not None else "__UNK__"
            cli_cat_vec.append(cat_maps[k].get(raw_key, 0))

        # Independent embryo type branch
        embryo_raw = cli.get(EMBRYO_TYPE_KEY, None)
        embryo_key = str(embryo_raw) if embryo_raw is not None else "__UNK__"
        embryo_idx = embryo_type_map.get(embryo_key, 0)

        image_arr.append(img_vec)
        doppler_arr.append(dop_vec)
        clinical_cont_arr.append(cli_cont_vec)
        clinical_cat_arr.append(cli_cat_vec)
        embryo_type_arr.append(embryo_idx)
        labels.append(safe_label(rec.get("label", None)))
        patient_ids.append(str(rec.get("patient_id", Path(path).stem)))
        file_paths.append(str(path))

    image_arr = np.array(image_arr, dtype=np.float32) if len(image_arr) else np.zeros((0, 0), dtype=np.float32)
    doppler_arr = np.array(doppler_arr, dtype=np.float32) if len(doppler_arr) else np.zeros((0, 0), dtype=np.float32)
    clinical_cont_arr = np.array(clinical_cont_arr, dtype=np.float32) if len(clinical_cont_arr) else np.zeros((0, 0), dtype=np.float32)
    clinical_cat_arr = np.array(clinical_cat_arr, dtype=np.int64) if len(clinical_cat_arr) else np.zeros((0, 0), dtype=np.int64)
    embryo_type_arr = np.array(embryo_type_arr, dtype=np.int64) if len(embryo_type_arr) else np.zeros((0,), dtype=np.int64)
    labels = np.array(labels, dtype=np.int64)

    return {
        "image": image_arr,
        "doppler": doppler_arr,
        "clinical_cont": clinical_cont_arr,
        "clinical_cat": clinical_cat_arr,
        "embryo_type": embryo_type_arr,
        "labels": labels,
        "patient_ids": patient_ids,
        "file_paths": file_paths,
    }


# =========================
# 4. Missing-value imputation and standardization
# =========================
def fit_imputer_scaler(train_arr: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Estimate mean and standard deviation on the training split:
    - first fill NaN values with column means
    - then apply z-score standardization
    """
    if train_arr.size == 0:
        return {
            "mean": np.zeros((0,), dtype=np.float32),
            "std": np.ones((0,), dtype=np.float32),
        }

    mean = np.nanmean(train_arr, axis=0)
    mean = np.where(np.isnan(mean), 0.0, mean)
    filled = np.where(np.isnan(train_arr), mean, train_arr)
    std = filled.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}


def transform_with_scaler(arr: np.ndarray, scaler: Dict[str, np.ndarray]) -> np.ndarray:
    if arr.size == 0:
        return arr.astype(np.float32)
    mean = scaler["mean"]
    std = scaler["std"]
    filled = np.where(np.isnan(arr), mean, arr)
    out = (filled - mean) / std
    return out.astype(np.float32)


# =========================
# 5. Dataset with tabular augmentation
# =========================
class FusionJsonDataset(Dataset):
    def __init__(
        self,
        data_dict: Dict[str, Any],
        is_train: bool = False,
        noise_std_image: float = 0.0,
        noise_std_doppler: float = 0.0,
        noise_std_clinical: float = 0.0,
        feature_dropout_prob: float = 0.0,
        scale_jitter_std: float = 0.0,
        modality_dropout_prob: float = 0.0,
    ):
        self.image = torch.tensor(data_dict["image"], dtype=torch.float32)
        self.doppler = torch.tensor(data_dict["doppler"], dtype=torch.float32)
        self.clinical_cont = torch.tensor(data_dict["clinical_cont"], dtype=torch.float32)
        self.clinical_cat = torch.tensor(data_dict["clinical_cat"], dtype=torch.long)
        self.embryo_type = torch.tensor(data_dict["embryo_type"], dtype=torch.long)
        self.labels = torch.tensor(data_dict["labels"], dtype=torch.float32)
        self.patient_ids = data_dict["patient_ids"]
        self.file_paths = data_dict["file_paths"]

        self.is_train = is_train
        self.noise_std_image = max(0.0, float(noise_std_image))
        self.noise_std_doppler = max(0.0, float(noise_std_doppler))
        self.noise_std_clinical = max(0.0, float(noise_std_clinical))
        self.feature_dropout_prob = min(1.0, max(0.0, float(feature_dropout_prob)))
        self.scale_jitter_std = max(0.0, float(scale_jitter_std))
        self.modality_dropout_prob = min(1.0, max(0.0, float(modality_dropout_prob)))

    def _add_noise(self, x: torch.Tensor, noise_std: float) -> torch.Tensor:
        if noise_std <= 0.0 or x.numel() == 0:
            return x
        return x + torch.randn_like(x) * noise_std

    def _feature_dropout(self, x: torch.Tensor) -> torch.Tensor:
        if self.feature_dropout_prob <= 0.0 or x.numel() == 0:
            return x
        keep_mask = (torch.rand_like(x) > self.feature_dropout_prob).float()
        return x * keep_mask

    def _scale_jitter(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale_jitter_std <= 0.0 or x.numel() == 0:
            return x
        scale = 1.0 + torch.randn(1, device=x.device, dtype=x.dtype) * self.scale_jitter_std
        return x * scale

    def _modality_dropout(self, x: torch.Tensor) -> torch.Tensor:
        if self.modality_dropout_prob <= 0.0 or x.numel() == 0:
            return x
        if torch.rand(1).item() < self.modality_dropout_prob:
            return torch.zeros_like(x)
        return x

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        image = self.image[idx]
        doppler = self.doppler[idx]
        clinical_cont = self.clinical_cont[idx]

        # Apply augmentation only during training. Validation and test samples
        # must remain deterministic so that metrics and interpretability exports
        # reflect the real input distribution.
        if self.is_train:
            image = self._modality_dropout(
                self._feature_dropout(
                    self._scale_jitter(
                        self._add_noise(image, self.noise_std_image)
                    )
                )
            )
            doppler = self._modality_dropout(
                self._feature_dropout(
                    self._scale_jitter(
                        self._add_noise(doppler, self.noise_std_doppler)
                    )
                )
            )
            clinical_cont = self._modality_dropout(
                self._feature_dropout(
                    self._scale_jitter(
                        self._add_noise(clinical_cont, self.noise_std_clinical)
                    )
                )
            )

        return {
            "image": image,
            "doppler": doppler,
            "clinical_cont": clinical_cont,
            "clinical_cat": self.clinical_cat[idx],
            "embryo_type": self.embryo_type[idx],
            "label": self.labels[idx],
            "patient_id": self.patient_ids[idx],
            "file_path": self.file_paths[idx],
        }


# =========================
# 6. Model modules
# =========================
class MLPBlock(nn.Module):
    """Reusable MLP encoder block used by numeric and branch encoders."""
    def __init__(self, in_dim: int, hidden_dims: List[int], out_dim: int, dropout: float = 0.3):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FeatureGate(nn.Module):
    """
    Sample-wise feature gate.

    Given a numeric feature vector x, the module predicts a same-sized gate in
    the range [0, 1]. The gated vector lets the network emphasize or suppress
    individual features per sample. These gates are exported as model-internal
    weighting signals for interpretation; they should not be treated as causal
    feature effects.
    """
    def __init__(self, dim: int, hidden_ratio: float = 0.5, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        if dim <= 0:
            self.net = None
        else:
            hidden_dim = max(8, int(dim * hidden_ratio))
            self.net = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, dim),
                nn.Sigmoid(),
            )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.net is None or x.numel() == 0:
            gate = torch.ones_like(x)
            return x, gate
        gate = self.net(x)
        x_out = x * gate
        return x_out, gate


class OptionalNumericEncoder(nn.Module):
    """
    Used for numeric vectors such as image, Doppler, and clinical_cont.
    Applies a feature gate before MLP encoding.
    """
    def __init__(self, in_dim: int, emb_dim: int, hidden_dims: List[int], dropout: float):
        super().__init__()
        self.in_dim = in_dim
        self.emb_dim = emb_dim
        if in_dim > 0:
            self.feature_gate = FeatureGate(in_dim, hidden_ratio=0.5, dropout=dropout * 0.5)
            self.encoder = MLPBlock(in_dim, hidden_dims, emb_dim, dropout)
        else:
            self.feature_gate = None
            self.encoder = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.encoder is None:
            out = torch.zeros((x.size(0), self.emb_dim), device=x.device, dtype=torch.float32)
            gate = torch.zeros((x.size(0), 0), device=x.device, dtype=torch.float32)
            return out, gate
        x_gated, gate = self.feature_gate(x)
        out = self.encoder(x_gated)
        return out, gate


class ClinicalEncoder(nn.Module):
    """
    Main clinical branch encoder:
    - continuous variables first pass through a feature gate
    - categorical variables are embedded
    - all parts are concatenated and passed into an MLP

    Notes:Embryo type is handled separately by EmbryoTypeEncoder.
    """
    def __init__(
        self,
        cont_dim: int,
        cat_cardinalities: List[int],
        emb_dim: int = 96,
        cat_emb_cap: int = 8,
        hidden_dims: List[int] = [192, 128],
        dropout: float = 0.3,
    ):
        super().__init__()
        self.cont_dim = cont_dim
        self.cat_cardinalities = cat_cardinalities

        # Feature gate for continuous clinical features
        self.cont_gate = FeatureGate(cont_dim, hidden_ratio=0.5, dropout=dropout * 0.5) if cont_dim > 0 else None

        # One embedding layer per categorical field
        self.cat_embeddings = nn.ModuleList()
        for card in cat_cardinalities:
            emb_size = min(cat_emb_cap, max(2, int(math.sqrt(card)) + 1))
            self.cat_embeddings.append(nn.Embedding(card, emb_size))

        total_cat_emb_dim = sum(emb.embedding_dim for emb in self.cat_embeddings)
        in_dim = cont_dim + total_cat_emb_dim
        self.encoder = MLPBlock(in_dim, hidden_dims, emb_dim, dropout) if in_dim > 0 else None
        self.emb_dim = emb_dim

    def forward(self, clinical_cont: torch.Tensor, clinical_cat: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        parts = []
        aux = {}

        # Continuous-variable block
        if self.cont_dim > 0:
            cont_gated, cont_gate = self.cont_gate(clinical_cont)
            parts.append(cont_gated)
            aux["clinical_cont_gate"] = cont_gate
        else:
            aux["clinical_cont_gate"] = torch.zeros((clinical_cont.size(0), 0), device=clinical_cont.device)

        # Categorical-variable block
        if len(self.cat_embeddings) > 0:
            cat_parts = []
            for i, emb in enumerate(self.cat_embeddings):
                cat_parts.append(emb(clinical_cat[:, i]))
            cat_embed = torch.cat(cat_parts, dim=1)
            parts.append(cat_embed)
        
        if len(parts) == 0:
            out = torch.zeros((clinical_cont.size(0), self.emb_dim), device=clinical_cont.device, dtype=torch.float32)
        else:
            x = torch.cat(parts, dim=1)
            out = self.encoder(x)
        return out, aux


class EmbryoTypeEncoder(nn.Module):
    """
    Independent embryo type encoder.
    Embryo-stage information is kept as an independent input,
    so it has its own branch instead of being merged into clinical_cat.
    """
    def __init__(self, num_embeddings: int, emb_dim: int = 96, inner_dim: int = 16, dropout: float = 0.2):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, inner_dim)
        self.proj = nn.Sequential(
            nn.Linear(inner_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, embryo_type: torch.Tensor) -> torch.Tensor:
        x = self.embedding(embryo_type)
        x = self.proj(x)
        return x


class TokenGate(nn.Module):
    """Modality-level gate that estimates how strongly a branch token should be trusted."""
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FourBranchGatedAttentionFusionModel(nn.Module):
    """
    Four-branch gated-attention fusion model.

    Branches:
    1. image/radiomics features
    2. Doppler/color-flow features
    3. clinical continuous and categorical variables
    4. embryo type

    Flow:
    raw features -> feature gates -> branch encoders -> modality gates ->
    multi-head token attention -> learned token pooling -> binary classifier.

    The forward pass also returns auxiliary tensors used for interpretation:
    feature-level gates, modality gates, attention weights, and final token
    pooling weights (`token_alpha`).
    """
    def __init__(
        self,
        image_dim: int,
        doppler_dim: int,
        clinical_cont_dim: int,
        clinical_cat_cardinalities: List[int],
        embryo_type_cardinality: int,
        emb_dim: int = 96,
        branch_hidden_dims: List[int] = [192, 128],
        fusion_hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.35,
        ffn_expand_ratio: int = 2,
    ):
        super().__init__()
        self.emb_dim = emb_dim

        # Encode each modality into the same embedding dimension. Keeping a
        # shared token width makes cross-modality attention straightforward.
        self.image_encoder = OptionalNumericEncoder(image_dim, emb_dim, branch_hidden_dims, dropout)
        self.doppler_encoder = OptionalNumericEncoder(doppler_dim, emb_dim, branch_hidden_dims, dropout)
        self.clinical_encoder = ClinicalEncoder(
            cont_dim=clinical_cont_dim,
            cat_cardinalities=clinical_cat_cardinalities,
            emb_dim=emb_dim,
            hidden_dims=branch_hidden_dims,
            dropout=dropout,
        )
        self.embryo_encoder = EmbryoTypeEncoder(
            num_embeddings=embryo_type_cardinality,
            emb_dim=emb_dim,
            inner_dim=min(16, max(4, emb_dim // 6)),
            dropout=dropout * 0.5,
        )

        # Modality gates condition each token on related tokens before attention.
        # Image and Doppler gates are guided by clinical and embryo information.
        self.img_gate = TokenGate(emb_dim * 3, emb_dim, fusion_hidden_dim, dropout)
        self.dop_gate = TokenGate(emb_dim * 3, emb_dim, fusion_hidden_dim, dropout)
        # The clinical gate can use information from all branches.
        self.cli_gate = TokenGate(emb_dim * 4, emb_dim, fusion_hidden_dim, dropout)
        # Embryo type is mainly modulated by itself and the clinical token.
        self.emb_gate = TokenGate(emb_dim * 2, emb_dim, fusion_hidden_dim, dropout)

        # Cross-modality attention lets each branch exchange information with
        # the other three branch tokens.
        self.modality_attn = nn.MultiheadAttention(
            embed_dim=emb_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(emb_dim)

        # Transformer-style feed-forward block after attention.
        ffn_hidden = emb_dim * max(1, int(ffn_expand_ratio))
        self.post_attn_ffn = nn.Sequential(
            nn.Linear(emb_dim, ffn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, emb_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(emb_dim)

        # Learned token pooling produces the final sample-specific contribution
        # weight for each modality.
        self.token_score = nn.Sequential(
            nn.Linear(emb_dim, emb_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(emb_dim // 2, 1),
        )

        # Binary classification head. The output is a single logit.
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, 1),
        )

    def forward(self, image, doppler, clinical_cont, clinical_cat, embryo_type):
        # 1) Encode each branch.
        h_img, image_feature_gate = self.image_encoder(image)
        h_dop, doppler_feature_gate = self.doppler_encoder(doppler)
        h_cli, cli_aux = self.clinical_encoder(clinical_cont, clinical_cat)
        h_emb = self.embryo_encoder(embryo_type)

        clinical_cont_gate = cli_aux["clinical_cont_gate"]

        # 2) Apply modality-level gates.
        g_img = self.img_gate(torch.cat([h_img, h_cli, h_emb], dim=1))
        g_dop = self.dop_gate(torch.cat([h_dop, h_cli, h_emb], dim=1))
        g_cli = self.cli_gate(torch.cat([h_cli, h_img, h_dop, h_emb], dim=1))
        g_emb = self.emb_gate(torch.cat([h_emb, h_cli], dim=1))

        h_img = h_img * g_img
        h_dop = h_dop * g_dop
        h_cli = h_cli * g_cli
        h_emb = h_emb * g_emb

        # 3) Run self-attention over the four modality tokens.
        tokens = torch.stack([h_img, h_dop, h_cli, h_emb], dim=1)  # [B, 4, D]
        attn_out, attn_weights = self.modality_attn(
            tokens, tokens, tokens,
            need_weights=True,
            average_attn_weights=False,
        )
        tokens = self.attn_norm(tokens + attn_out)
        tokens = self.ffn_norm(tokens + self.post_attn_ffn(tokens))

        # 4) Pool tokens with learned per-sample modality weights.
        token_logits = self.token_score(tokens).squeeze(-1)    # [B, 4]
        token_alpha = torch.softmax(token_logits, dim=1)       # [B, 4]
        fused = torch.sum(tokens * token_alpha.unsqueeze(-1), dim=1)  # [B, D]

        # 5) Predict the binary outcome.
        logits = self.classifier(fused).squeeze(1)

        # attn_weights is usually [B, num_heads, T, S], with T=S=4. Reporting
        # code can average over the batch/head axes for compact summaries.
        aux = {
            "image_feature_gate": image_feature_gate,
            "doppler_feature_gate": doppler_feature_gate,
            "clinical_cont_gate": clinical_cont_gate,
            "g_img": g_img,
            "g_dop": g_dop,
            "g_cli": g_cli,
            "g_emb": g_emb,
            "attn_weights": attn_weights,
            "token_alpha": token_alpha,
            "h_img": h_img,
            "h_dop": h_dop,
            "h_cli": h_cli,
            "h_emb": h_emb,
            "fused": fused,
        }
        return logits, aux


# =========================
# 7. Evaluation metrics
# =========================
def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(np.int64)

    metrics = {}
    try:
        if len(np.unique(y_true)) < 2:
            metrics["auc"] = float("nan")
        else:
            metrics["auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        metrics["auc"] = float("nan")

    metrics["acc"] = float(accuracy_score(y_true, y_pred))
    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    else:
        specificity = float("nan")
        sensitivity = float("nan")
        tn = fp = fn = tp = 0

    metrics["specificity"] = float(specificity)
    metrics["sensitivity"] = float(sensitivity)
    metrics["tn"] = int(tn)
    metrics["fp"] = int(fp)
    metrics["fn"] = int(fn)
    metrics["tp"] = int(tp)
    return metrics


def select_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    max_candidates: int = 256,
) -> Tuple[float, Dict[str, float]]:
    """
    Select a validation-set operating threshold.

    The score prioritizes accuracy, then uses F1 and sensitivity as secondary
    terms. This threshold is useful for reporting but should be selected only on
    validation data, never on the held-out test split.
    """
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=np.float64).reshape(-1)
    if y_true.size == 0:
        return 0.5, compute_metrics(y_true, y_prob, threshold=0.5)

    if np.allclose(y_prob, y_prob[0]):
        m = compute_metrics(y_true, y_prob, threshold=0.5)
        return 0.5, m

    uniq = np.unique(y_prob)
    if uniq.size > max_candidates:
        q = np.linspace(0.01, 0.99, max_candidates)
        cand = np.unique(np.quantile(uniq, q))
    else:
        cand = uniq
    cand = np.unique(np.concatenate([[0.5], cand]))

    best_thr = 0.5
    best_metrics = compute_metrics(y_true, y_prob, threshold=best_thr)
    best_score = best_metrics["acc"] + 0.20 * best_metrics["f1"] + 0.05 * best_metrics["sensitivity"]

    for thr in cand:
        m = compute_metrics(y_true, y_prob, threshold=float(thr))
        score = m["acc"] + 0.20 * m["f1"] + 0.05 * m["sensitivity"]
        if score > best_score:
            best_score = score
            best_thr = float(thr)
            best_metrics = m

    return best_thr, best_metrics


def apply_tabular_mixup(
    image: torch.Tensor,
    doppler: torch.Tensor,
    clinical_cont: torch.Tensor,
    clinical_cat: torch.Tensor,
    embryo_type: torch.Tensor,
    labels: torch.Tensor,
    mixup_alpha: float,
    mixup_prob: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply lightweight mixup to reduce overfitting on small tabular datasets."""
    batch_size = labels.size(0)
    if batch_size < 2 or mixup_prob <= 0.0 or mixup_alpha <= 0.0:
        return image, doppler, clinical_cont, clinical_cat, embryo_type, labels
    if torch.rand(1, device=labels.device).item() >= mixup_prob:
        return image, doppler, clinical_cont, clinical_cat, embryo_type, labels

    lam = float(np.random.beta(mixup_alpha, mixup_alpha))
    lam = max(lam, 1.0 - lam)
    index = torch.randperm(batch_size, device=labels.device)

    image = lam * image + (1.0 - lam) * image[index]
    doppler = lam * doppler + (1.0 - lam) * doppler[index]
    clinical_cont = lam * clinical_cont + (1.0 - lam) * clinical_cont[index]

    # Categorical branches cannot be linearly interpolated, so we stochastically swap categories.
    swap_mask = torch.rand(batch_size, device=labels.device) > lam
    clinical_cat = torch.where(swap_mask.unsqueeze(1), clinical_cat[index], clinical_cat)
    embryo_type = torch.where(swap_mask, embryo_type[index], embryo_type)

    labels = lam * labels + (1.0 - lam) * labels[index]
    return image, doppler, clinical_cont, clinical_cat, embryo_type, labels


# =========================
# 8. Single-epoch training or validation
# =========================
def run_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    train: bool = True,
    label_smoothing: float = 0.0,
    mixup_prob: float = 0.0,
    mixup_alpha: float = 0.0,
    collect_aux: bool = False,
):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    all_probs = []
    all_labels = []
    all_patient_ids = []

    aux_bucket = {
        "image_feature_gate": [],
        "doppler_feature_gate": [],
        "clinical_cont_gate": [],
        "token_alpha": [],
        "attn_weights": [],
        "g_img": [],
        "g_dop": [],
        "g_cli": [],
        "g_emb": [],
    }

    for batch in loader:
        image = batch["image"].to(device)
        doppler = batch["doppler"].to(device)
        clinical_cont = batch["clinical_cont"].to(device)
        clinical_cat = batch["clinical_cat"].to(device)
        embryo_type = batch["embryo_type"].to(device)
        labels = batch["label"].to(device)

        if train:
            image, doppler, clinical_cont, clinical_cat, embryo_type, labels = apply_tabular_mixup(
                image=image,
                doppler=doppler,
                clinical_cont=clinical_cont,
                clinical_cat=clinical_cat,
                embryo_type=embryo_type,
                labels=labels,
                mixup_alpha=mixup_alpha,
                mixup_prob=mixup_prob,
            )

        with torch.set_grad_enabled(train):
            logits, aux = model(image, doppler, clinical_cont, clinical_cat, embryo_type)

            # Optional label smoothing
            targets = labels
            if train and label_smoothing > 0.0:
                targets = labels * (1.0 - label_smoothing) + 0.5 * label_smoothing

            loss = criterion(logits, targets)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        probs = torch.sigmoid(logits)
        total_loss += loss.item() * labels.size(0)
        all_probs.append(probs.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())
        all_patient_ids.extend(batch["patient_id"])

        if collect_aux:
            for k in aux_bucket.keys():
                if k in aux:
                    aux_bucket[k].append(aux[k].detach().cpu())

    y_prob = np.concatenate(all_probs) if all_probs else np.array([])
    y_true = np.concatenate(all_labels) if all_labels else np.array([])
    avg_loss = total_loss / max(1, len(loader.dataset))
    metrics = compute_metrics(y_true.astype(np.int64), y_prob.astype(np.float64))

    merged_aux = None
    if collect_aux:
        merged_aux = {}
        for k, vals in aux_bucket.items():
            if len(vals) == 0:
                merged_aux[k] = None
            else:
                merged_aux[k] = torch.cat(vals, dim=0)

    return avg_loss, metrics, y_true, y_prob, all_patient_ids, merged_aux


# =========================
# 9. Result saving
# =========================
def save_predictions_csv(
    output_path: str,
    patient_ids: List[str],
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
):
    import csv
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["patient_id", "label", "prob", "pred", "threshold"])
        for pid, yt, yp in zip(patient_ids, y_true, y_prob):
            writer.writerow([pid, int(yt), float(yp), int(yp >= threshold), float(threshold)])


def serialize_meta(meta: Dict[str, Any], scalers: Dict[str, Any], output_path: str):
    obj = {
        "feature_meta": meta,
        "scalers": {
            k: {kk: vv.tolist() for kk, vv in v.items()} for k, v in scalers.items()
        },
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _finite_1d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64).reshape(-1)
    return arr[np.isfinite(arr)]


def _compute_psi(train_col: np.ndarray, val_col: np.ndarray, n_bins: int = 10) -> float:
    tr = _finite_1d(train_col)
    va = _finite_1d(val_col)
    if tr.size < 10 or va.size < 10:
        return float("nan")

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(tr, quantiles)
    edges = np.unique(edges)
    if edges.size < 3:
        return float("nan")

    tr_hist, _ = np.histogram(tr, bins=edges)
    va_hist, _ = np.histogram(va, bins=edges)

    eps = 1e-6
    tr_pct = (tr_hist + eps) / (tr_hist.sum() + eps * tr_hist.size)
    va_pct = (va_hist + eps) / (va_hist.sum() + eps * va_hist.size)
    psi = np.sum((va_pct - tr_pct) * np.log(va_pct / tr_pct))
    return float(psi)


def _numeric_drift_rows(feature_names: List[str], train_arr: np.ndarray, val_arr: np.ndarray, group_name: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if train_arr.ndim != 2 or val_arr.ndim != 2:
        return rows
    if train_arr.shape[1] == 0 or val_arr.shape[1] == 0:
        return rows

    feat_dim = min(train_arr.shape[1], val_arr.shape[1], len(feature_names))
    for i in range(feat_dim):
        tr = _finite_1d(train_arr[:, i])
        va = _finite_1d(val_arr[:, i])
        if tr.size == 0 or va.size == 0:
            continue

        tr_mean = float(np.mean(tr))
        va_mean = float(np.mean(va))
        tr_std = float(np.std(tr))
        va_std = float(np.std(va))

        mean_shift_z = abs(va_mean - tr_mean) / (tr_std + 1e-8)
        std_ratio = va_std / (tr_std + 1e-8)
        psi = _compute_psi(tr, va, n_bins=10)

        rows.append(
            {
                "group": group_name,
                "feature_name": feature_names[i],
                "train_mean": tr_mean,
                "val_mean": va_mean,
                "train_std": tr_std,
                "val_std": va_std,
                "mean_shift_z": float(mean_shift_z),
                "std_ratio": float(std_ratio),
                "psi": psi,
            }
        )
    return rows


def _categorical_tv_distance(train_ids: np.ndarray, val_ids: np.ndarray, n_class: int) -> float:
    if n_class <= 0:
        return float("nan")
    tr = np.asarray(train_ids, dtype=np.int64).reshape(-1)
    va = np.asarray(val_ids, dtype=np.int64).reshape(-1)
    tr = tr[(tr >= 0) & (tr < n_class)]
    va = va[(va >= 0) & (va < n_class)]
    if tr.size == 0 or va.size == 0:
        return float("nan")

    tr_freq = np.bincount(tr, minlength=n_class).astype(np.float64)
    va_freq = np.bincount(va, minlength=n_class).astype(np.float64)
    tr_freq /= max(1.0, tr_freq.sum())
    va_freq /= max(1.0, va_freq.sum())
    return float(0.5 * np.abs(tr_freq - va_freq).sum())


def analyze_train_val_drift(
    train_data_raw: Dict[str, Any],
    val_data_raw: Dict[str, Any],
    meta: Dict[str, Any],
    output_dir: str,
) -> Dict[str, Any]:
    import csv

    numeric_rows: List[Dict[str, Any]] = []
    numeric_rows.extend(_numeric_drift_rows(meta["image_keys"], train_data_raw["image"], val_data_raw["image"], "image"))
    numeric_rows.extend(_numeric_drift_rows(meta["doppler_keys"], train_data_raw["doppler"], val_data_raw["doppler"], "doppler"))
    numeric_rows.extend(
        _numeric_drift_rows(
            meta["clinical_cont_keys"],
            train_data_raw["clinical_cont"],
            val_data_raw["clinical_cont"],
            "clinical_cont",
        )
    )

    # Common heuristics: PSI > 0.25 suggests notable drift; mean_shift_z > 0.8 suggests a large mean shift.
    large_psi_count = sum(1 for r in numeric_rows if np.isfinite(r["psi"]) and r["psi"] > 0.25)
    large_shift_count = sum(1 for r in numeric_rows if np.isfinite(r["mean_shift_z"]) and r["mean_shift_z"] > 0.8)
    abnormal_std_ratio_count = sum(1 for r in numeric_rows if np.isfinite(r["std_ratio"]) and (r["std_ratio"] < 0.67 or r["std_ratio"] > 1.5))

    numeric_rows_sorted = sorted(
        numeric_rows,
        key=lambda x: (
            x["psi"] if np.isfinite(x["psi"]) else -1e9,
            x["mean_shift_z"] if np.isfinite(x["mean_shift_z"]) else -1e9,
        ),
        reverse=True,
    )
    top_numeric_drift = numeric_rows_sorted[:20]

    categorical_rows: List[Dict[str, Any]] = []
    for i, k in enumerate(meta["clinical_cat_keys"]):
        card = len(meta["cat_maps"][k])
        tv = _categorical_tv_distance(train_data_raw["clinical_cat"][:, i], val_data_raw["clinical_cat"][:, i], card)
        categorical_rows.append({"feature_name": k, "group": "clinical_cat", "cardinality": card, "tv_distance": tv})

    embryo_card = len(meta["embryo_type_map"])
    embryo_tv = _categorical_tv_distance(train_data_raw["embryo_type"], val_data_raw["embryo_type"], embryo_card)
    categorical_rows.append(
        {
            "feature_name": meta["embryo_type_key"],
            "group": "embryo_type",
            "cardinality": embryo_card,
            "tv_distance": embryo_tv,
        }
    )
    large_tv_count = sum(1 for r in categorical_rows if np.isfinite(r["tv_distance"]) and r["tv_distance"] > 0.2)

    numeric_csv = os.path.join(output_dir, "train_val_drift_numeric.csv")
    with open(numeric_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["group", "feature_name", "train_mean", "val_mean", "train_std", "val_std", "mean_shift_z", "std_ratio", "psi"],
        )
        writer.writeheader()
        for r in numeric_rows:
            writer.writerow(r)

    categorical_csv = os.path.join(output_dir, "train_val_drift_categorical.csv")
    with open(categorical_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["group", "feature_name", "cardinality", "tv_distance"])
        writer.writeheader()
        for r in categorical_rows:
            writer.writerow(r)

    summary = {
        "n_train": int(len(train_data_raw["labels"])),
        "n_val": int(len(val_data_raw["labels"])),
        "numeric_feature_count": int(len(numeric_rows)),
        "categorical_feature_count": int(len(categorical_rows)),
        "large_psi_count": int(large_psi_count),
        "large_mean_shift_count": int(large_shift_count),
        "abnormal_std_ratio_count": int(abnormal_std_ratio_count),
        "large_categorical_tv_count": int(large_tv_count),
        "top_numeric_drift": top_numeric_drift,
    }

    summary_json = os.path.join(output_dir, "train_val_drift_summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[Info] Data drift analysis completed")
    print(
        f"[Info] Numeric features total={len(numeric_rows)}, PSI>0.25={large_psi_count}, "
        f"mean_shift_z>0.8={large_shift_count}, abnormal std_ratio={abnormal_std_ratio_count}"
    )
    print(f"[Info] Categorical features: total={len(categorical_rows)}, TV>0.20={large_tv_count}")
    print(f"[Info] Drift report saved to: {summary_json}")
    return summary


def save_feature_importance_csv(output_path: str, names: List[str], mean_values: np.ndarray, std_values: np.ndarray):
    import csv
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["feature_name", "mean_gate", "std_gate"])
        for n, m, s in zip(names, mean_values.tolist(), std_values.tolist()):
            writer.writerow([n, float(m), float(s)])


def save_token_importance_csv(output_path: str, token_alpha: torch.Tensor):
    import csv
    alpha_np = token_alpha.numpy()  # [N, 4]
    mean_alpha = alpha_np.mean(axis=0)
    std_alpha = alpha_np.std(axis=0)
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["token_name", "mean_weight", "std_weight"])
        for name, m, s in zip(TOKEN_NAMES, mean_alpha.tolist(), std_alpha.tolist()):
            writer.writerow([name, float(m), float(s)])


def save_attention_matrix_json(output_path: str, attn_weights: torch.Tensor):
    """
    Save the average attention matrix between the four tokens.
    Expected attn_weights shape: [N, H, T, S]
    """
    if attn_weights is None:
        return
    attn_np = attn_weights.numpy()
    mean_attn = attn_np.mean(axis=(0, 1))  # [T, S]
    obj = {
        "token_names": TOKEN_NAMES,
        "mean_attention_matrix": mean_attn.tolist(),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def export_interpretability(
    aux: Dict[str, torch.Tensor],
    meta: Dict[str, Any],
    output_dir: str,
):
    """
    Export validation-set interpretability outputs:
    1) image_feature_gate_summary.csv
    2) doppler_feature_gate_summary.csv
    3) clinical_cont_gate_summary.csv
    4) token_importance_summary.csv
    5) attention_matrix_summary.json
    """
    if aux is None:
        return

    # Image feature gates
    if aux.get("image_feature_gate") is not None and aux["image_feature_gate"].numel() > 0:
        g = aux["image_feature_gate"].numpy()
        save_feature_importance_csv(
            os.path.join(output_dir, "image_feature_gate_summary.csv"),
            meta["image_keys"],
            g.mean(axis=0),
            g.std(axis=0),
        )

    # Doppler feature gates
    if aux.get("doppler_feature_gate") is not None and aux["doppler_feature_gate"].numel() > 0:
        g = aux["doppler_feature_gate"].numpy()
        save_feature_importance_csv(
            os.path.join(output_dir, "doppler_feature_gate_summary.csv"),
            meta["doppler_keys"],
            g.mean(axis=0),
            g.std(axis=0),
        )

    # Clinical continuous feature gates
    if aux.get("clinical_cont_gate") is not None and aux["clinical_cont_gate"].numel() > 0:
        g = aux["clinical_cont_gate"].numpy()
        save_feature_importance_csv(
            os.path.join(output_dir, "clinical_cont_gate_summary.csv"),
            meta["clinical_cont_keys"],
            g.mean(axis=0),
            g.std(axis=0),
        )

    # Final fusion weights for the four tokens
    if aux.get("token_alpha") is not None:
        save_token_importance_csv(
            os.path.join(output_dir, "token_importance_summary.csv"),
            aux["token_alpha"],
        )

    # Average inter-token attention matrix
    if aux.get("attn_weights") is not None:
        save_attention_matrix_json(
            os.path.join(output_dir, "attention_matrix_summary.json"),
            aux["attn_weights"],
        )


# =========================
# 10. Main workflow
# =========================
def main(args):
    configure_utf8_stdio()
    set_seed(args.seed)
    ensure_dir(args.output_dir)

    train_jsons = find_json_files(args.train_dir)
    val_jsons = find_json_files(args.val_dir)

    if len(train_jsons) == 0:
        raise RuntimeError(f"No JSON files found under train_dir: {args.train_dir}")
    if len(val_jsons) == 0:
        raise RuntimeError(f"No JSON files found under val_dir: {args.val_dir}")

    print(f"[Info] train JSON count: {len(train_jsons)}")
    print(f"[Info] val JSON count: {len(val_jsons)}")

    # ===== Build feature space =====
    meta = build_feature_spaces(train_jsons, val_jsons)
    print(f"[Info] clinical cont dim : {len(meta['clinical_cont_keys'])}")
    print(f"[Info] clinical cat dim  : {len(meta['clinical_cat_keys'])}")
    print(f"[Info] image feat dim    : {len(meta['image_keys'])}")
    print(f"[Info] doppler feat dim  : {len(meta['doppler_keys'])}")
    print(f"[Info] embryo type class : {len(meta['embryo_type_map'])}")

    # ===== Convert to arrays =====
    train_data_raw = build_array_and_labels(train_jsons, meta)
    val_data_raw = build_array_and_labels(val_jsons, meta)

    # ===== train Fit scalers on training data to avoid leakage =====
    scalers = {
        "image": fit_imputer_scaler(train_data_raw["image"]),
        "doppler": fit_imputer_scaler(train_data_raw["doppler"]),
        "clinical_cont": fit_imputer_scaler(train_data_raw["clinical_cont"]),
    }

    train_data = {
        **train_data_raw,
        "image": transform_with_scaler(train_data_raw["image"], scalers["image"]),
        "doppler": transform_with_scaler(train_data_raw["doppler"], scalers["doppler"]),
        "clinical_cont": transform_with_scaler(train_data_raw["clinical_cont"], scalers["clinical_cont"]),
    }
    val_data = {
        **val_data_raw,
        "image": transform_with_scaler(val_data_raw["image"], scalers["image"]),
        "doppler": transform_with_scaler(val_data_raw["doppler"], scalers["doppler"]),
        "clinical_cont": transform_with_scaler(val_data_raw["clinical_cont"], scalers["clinical_cont"]),
    }

    analyze_train_val_drift(train_data_raw, val_data_raw, meta, args.output_dir)

    serialize_meta(meta, scalers, os.path.join(args.output_dir, "feature_meta_and_scalers.json"))

    # ===== Dataset / DataLoader =====
    train_dataset = FusionJsonDataset(
        train_data,
        is_train=True,
        noise_std_image=args.noise_std_image,
        noise_std_doppler=args.noise_std_doppler,
        noise_std_clinical=args.noise_std_clinical,
        feature_dropout_prob=args.feature_dropout_prob,
        scale_jitter_std=args.scale_jitter_std,
        modality_dropout_prob=args.modality_dropout_prob,
    )
    val_dataset = FusionJsonDataset(val_data, is_train=False)

    print(
        f"[Info] Training noise augmentation: image_std={args.noise_std_image}, "
        f"doppler_std={args.noise_std_doppler}, clinical_std={args.noise_std_clinical}, "
        f"feature_dropout_prob={args.feature_dropout_prob}, "
        f"scale_jitter_std={args.scale_jitter_std}, modality_dropout_prob={args.modality_dropout_prob}"
    )
    print(f"[Info] Training mixup augmentation: mixup_prob={args.mixup_prob}, mixup_alpha={args.mixup_alpha}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ===== Prepare model parameters =====
    clinical_cat_cardinalities = [len(meta["cat_maps"][k]) for k in meta["clinical_cat_keys"]]
    embryo_type_cardinality = len(meta["embryo_type_map"])

    # ===== Select device automatically =====
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"[Info] Using device: {device}")
    print(f"[Info] Label smoothing: {args.label_smoothing}")
    print("[Info] Model selection: validation AUC first; higher ACC breaks AUC ties")

    model = FourBranchGatedAttentionFusionModel(
        image_dim=train_data["image"].shape[1],
        doppler_dim=train_data["doppler"].shape[1],
        clinical_cont_dim=train_data["clinical_cont"].shape[1],
        clinical_cat_cardinalities=clinical_cat_cardinalities,
        embryo_type_cardinality=embryo_type_cardinality,
        emb_dim=args.emb_dim,
        branch_hidden_dims=args.branch_hidden_dims,
        fusion_hidden_dim=args.fusion_hidden_dim,
        num_heads=args.num_heads,
        dropout=args.dropout,
        ffn_expand_ratio=args.ffn_expand_ratio,
    ).to(device)

    # ===== Class imbalance: compute pos_weight automatically =====
    y_train = train_data["labels"]
    pos = float((y_train == 1).sum())
    neg = float((y_train == 0).sum())
    if pos > 0 and neg > 0:
        pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"[Info] BCEWithLogitsLoss pos_weight = {pos_weight.item():.4f}")
    else:
        criterion = nn.BCEWithLogitsLoss()
        print("[Warn] Training split contains only one class; pos_weight is disabled")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ReduceLROnPlateau is a stable default for small structured datasets.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=args.min_lr,
    )
    print("[Info] Learning-rate scheduler: ReduceLROnPlateau")

    history = []
    best_auc = -1e9
    best_acc_when_best_auc = -1e9
    best_epoch = -1
    patience_counter = 0
    best_model_path = os.path.join(args.output_dir, "best_model.pt")

    # ===== Training loop =====
    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics, _, _, _, _ = run_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            train=True,
            label_smoothing=args.label_smoothing,
            mixup_prob=args.mixup_prob,
            mixup_alpha=args.mixup_alpha,
            collect_aux=False,
        )
        val_loss, val_metrics, val_y_true, val_y_prob, val_patient_ids, _ = run_one_epoch(
            model,
            val_loader,
            optimizer,
            criterion,
            device,
            train=False,
            mixup_prob=0.0,
            mixup_alpha=0.0,
            collect_aux=False,
        )

        val_auc = val_metrics["auc"]
        val_acc = val_metrics["acc"]

        # AUC is primary; ACC is the secondary tie-breaker
        select_score = val_auc if not math.isnan(val_auc) else -1e9
        scheduler.step(select_score if select_score > -1e8 else val_acc)

        row = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": float(train_loss),
            **{f"train_{k}": v for k, v in train_metrics.items()},
            "val_loss": float(val_loss),
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "val_best_threshold": 0.5,
            "val_opt_acc": float(val_acc),
        }
        history.append(row)

        print(
            f"Epoch [{epoch:03d}/{args.epochs:03d}] | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
            f"train_auc={train_metrics['auc']:.4f} val_auc={val_metrics['auc']:.4f} | "
            f"train_acc={train_metrics['acc']:.4f} val_acc@0.5={val_metrics['acc']:.4f} "
            f"thr=0.5000 | "
            f"val_f1={val_metrics['f1']:.4f} val_sens={val_metrics['sensitivity']:.4f} val_spec={val_metrics['specificity']:.4f}"
        )

        updated_best = False
        if math.isnan(val_auc):
            # If AUC cannot be computed, select by ACC.
            if val_acc > best_acc_when_best_auc:
                updated_best = True
        else:
            if (val_auc > best_auc) or (abs(val_auc - best_auc) < 1e-12 and val_acc > best_acc_when_best_auc):
                updated_best = True

        if updated_best:
            if not math.isnan(val_auc):
                best_auc = float(val_auc)
            best_acc_when_best_auc = float(val_acc)
            best_epoch = epoch
            patience_counter = 0

            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "best_auc": best_auc,
                "best_acc_when_best_auc": best_acc_when_best_auc,
                "meta": meta,
                "scalers": {
                    "image": {k: v.tolist() for k, v in scalers["image"].items()},
                    "doppler": {k: v.tolist() for k, v in scalers["doppler"].items()},
                    "clinical_cont": {k: v.tolist() for k, v in scalers["clinical_cont"].items()},
                },
                "args": vars(args),
            }, best_model_path)

            save_predictions_csv(
                os.path.join(args.output_dir, "best_val_predictions.csv"),
                val_patient_ids,
                val_y_true,
                val_y_prob,
                threshold=0.5,
            )
        else:
            patience_counter += 1

        with open(os.path.join(args.output_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        if patience_counter >= args.patience:
            print(f"[Info] Early stopping triggered, best_epoch = {best_epoch}, best_auc = {best_auc:.6f}")
            break

    # ===== Reload the best model and export validation-set interpretation outputs =====
    ckpt = torch.load(best_model_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    _, best_val_metrics, best_val_y_true, best_val_y_prob, best_val_patient_ids, best_val_aux = run_one_epoch(
        model,
        val_loader,
        optimizer=None,
        criterion=criterion,
        device=device,
        train=False,
        mixup_prob=0.0,
        mixup_alpha=0.0,
        collect_aux=True,
    )
    save_predictions_csv(
        os.path.join(args.output_dir, "best_val_predictions.csv"),
        best_val_patient_ids,
        best_val_y_true,
        best_val_y_prob,
        threshold=0.5,
    )

    export_interpretability(best_val_aux, meta, args.output_dir)

    print("=" * 90)
    print(f"Training completed, best epoch: {best_epoch}")
    print(f"Best validation AUC : {best_val_metrics['auc']:.6f}")
    print(f"Best validation ACC(0.5threshold) : {best_val_metrics['acc']:.6f}")
    print("Best validationthreshold: 0.500000")
    print(f"Best validation F1  : {best_val_metrics['f1']:.6f}")
    print(f"Best validation Sens: {best_val_metrics['sensitivity']:.6f}")
    print(f"Best validation Spec: {best_val_metrics['specificity']:.6f}")
    print(f"Best model saved to: {best_model_path}")
    print(f"Training history saved to: {os.path.join(args.output_dir, 'history.json')}")
    print(f"Best validation predictions saved to: {os.path.join(args.output_dir, 'best_val_predictions.csv')}")
    print(f"Interpretability outputs saved to: {args.output_dir}")


# =========================
# 11. Default parameters
# =========================
TRAIN_PARAMS = {
    # Data paths
    "train_dir": "data/folds/fold_1/train",
    "val_dir": "data/folds/fold_1/val",
    "output_dir": "runs/gated_fusion",
    # Training hyperparameters
    "epochs": 320,
    "batch_size": 32,
    "lr": 1.0e-4,
    "weight_decay": 2.0e-4,
    "patience": 90,
    "num_workers": 0,
    "seed": 42,
    "device": "auto",
    # Model architecture
    "emb_dim": 160,
    "fusion_hidden_dim": 192,
    "num_heads": 8,
    "dropout": 0.32,
    "branch_hidden_dims": [256, 128],
    "ffn_expand_ratio": 2,
    # Learning-rate schedule
    "lr_factor": 0.5,
    "lr_patience": 10,
    "min_lr": 1e-6,
    # Data augmentation
    "noise_std_image": 0.04,
    "noise_std_doppler": 0.03,
    "noise_std_clinical": 0.015,
    "feature_dropout_prob": 0.05,
    "scale_jitter_std": 0.015,
    "modality_dropout_prob": 0.02,
    "mixup_prob": 0.15,
    "mixup_alpha": 0.25,
    # Other settings
    "label_smoothing": 0.0,
}


def load_runtime_train_params() -> Dict[str, Any]:
    params = dict(TRAIN_PARAMS)
    override_path = os.environ.get("TRAIN_PARAMS_JSON", "").strip()
    if override_path:
        with open(override_path, "r", encoding="utf-8") as f:
            overrides = json.load(f)
        if not isinstance(overrides, dict):
            raise RuntimeError(f"TRAIN_PARAMS_JSON must point to a json object, got: {type(overrides)}")
        params.update(overrides)
    return params


if __name__ == "__main__":
    main(SimpleNamespace(**load_runtime_train_params()))
