# Data Format

This repository does not include patient data. To run the code, provide one JSON
file per sample under split directories such as:

```text
data/folds/fold_1/
  train/
    0/*.json
    1/*.json
  val/
    0/*.json
    1/*.json
  test/
    0/*.json
    1/*.json
```

The directory labels are optional for the loader. The authoritative label is the
top-level `label` field in each JSON file.

## Minimal Sample

```json
{
  "patient_id": "sample_001",
  "label": 1,
  "features": {
    "original_shape2D_MajorAxisLength": 12.3,
    "original_glrlm_GrayLevelNonUniformity": 45.6,
    "red_pixels": 120,
    "blue_pixels": 80,
    "color_pixel_ratio": 0.18
  },
  "clinical_info": {
    "age": 34,
    "uterine_position": "anteverted",
    "endometrial_thickness_mm": 9.2,
    "endometrial_pattern": "A",
    "antegrade_peristalsis": "mild",
    "peristalsis_direction": "forward",
    "endometrial_volume_ml": 3.1,
    "endometrial_blood_flow_sd": 2.4,
    "endometrial_blood_flow_pi": 1.1,
    "endometrial_blood_flow_ri": 0.6,
    "vascularization_index": 4.5,
    "flow_index": 28.2,
    "vascularization_flow_index": 1.3,
    "maternal_bmi": 21.8,
    "infertility_duration_years": 2.0,
    "embryo_type": "blastocyst"
  }
}
```

## Feature Routing

Image features are detected by the `original_` prefix. All remaining keys inside
`features` are routed to the Doppler branch. Clinical categorical fields are
embedded; clinical continuous fields are imputed and standardized.

Category vocabularies are built from the training split only. Unseen validation
or test categories are mapped to `__UNK__`.
