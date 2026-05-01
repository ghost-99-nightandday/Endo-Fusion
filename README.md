# EndoFusion: Four-Branch Gated-Attention Training Code

This repository contains the minimal training code for a PyTorch multimodal
binary classifier. The model uses four input branches:

- radiomics/image features
- Doppler or color-flow features
- clinical variables
- embryo type

No patient data, fold files, model weights, or generated results are included.

## Files

| File | Purpose |
| --- | --- |
| `train.py` | Data loading, preprocessing, model definition, training loop, and checkpoint saving. |
| `docs/DATA_FORMAT.md` | Expected JSON structure for each sample. |
| `examples/train_params.example.json` | Example training configuration. |
| `requirements.txt` | Minimal Python dependencies. |

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Install a PyTorch build that matches your own CPU/CUDA environment if needed.

## Data Layout

Prepare one JSON file per sample:

```text
data/folds/fold_1/
  train/
    0/*.json
    1/*.json
  val/
    0/*.json
    1/*.json
```

The training script reads JSON files recursively. The true label must be stored
in the top-level `label` field. See `docs/DATA_FORMAT.md` for the schema.

## Train

Copy the example config and edit the paths or training options:

```bash
copy examples\train_params.example.json train_params.local.json
set TRAIN_PARAMS_JSON=train_params.local.json
python train.py
```

By default, outputs are written to:

```text
runs/gated_fusion/
```

The main checkpoint is:

```text
runs/gated_fusion/best_model.pt
```

## Private Files

The `.gitignore` excludes local data, checkpoints, and training outputs. Before
publishing to GitHub, check that only code and documentation are staged.
