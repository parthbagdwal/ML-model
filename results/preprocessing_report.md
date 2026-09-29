# mineRakshak-ai: Stage 4 Preprocessing & Splitting Report

> **DATA LEAKAGE PREVENTION CERTIFICATION:**
> The dataset was partitioned into Train, Validation, and Test sets PRIOR to fitting any transformers.
> The One-Hot categorical encoder was fitted EXCLUSIVELY on `X_train`.
> Validation and Test partitions were transformed using the pre-fitted transformer.

---

## 1. Input Dataset Overview

* **Source File**: `data/raw/synthetic_mine_data.csv`
* **Total Rows**: `20,000`
* **Total Raw Features**: `12` (`11` numerical + `1` categorical)
* **Target Column**: `risk_level`

## 2. Feature Schema & Engineering Rationale

| Feature | Type | Preprocessing Action | Rationale |
| :--- | :--- | :--- | :--- |
| `distance_m` | Numerical (float) | Pass-through (no scaling) | Tree-based models (XGBoost/HGB) are invariant to monotonic scaling; preserves physical units. Physical sensor attribute. |
| `object_x_m` | Numerical (float) | Pass-through (no scaling) | Tree-based models (XGBoost/HGB) are invariant to monotonic scaling; preserves physical units. Physical sensor attribute. |
| `object_y_m` | Numerical (float) | Pass-through (no scaling) | Tree-based models (XGBoost/HGB) are invariant to monotonic scaling; preserves physical units. Physical sensor attribute. |
| `object_z_m` | Numerical (float) | Pass-through (no scaling) | Tree-based models (XGBoost/HGB) are invariant to monotonic scaling; preserves physical units. Physical sensor attribute. |
| `object_width_m` | Numerical (float) | Pass-through (no scaling) | Tree-based models (XGBoost/HGB) are invariant to monotonic scaling; preserves physical units. Physical sensor attribute. |
| `object_height_m` | Numerical (float) | Pass-through (no scaling) | Tree-based models (XGBoost/HGB) are invariant to monotonic scaling; preserves physical units. Physical sensor attribute. |
| `object_length_m` | Numerical (float) | Pass-through (no scaling) | Tree-based models (XGBoost/HGB) are invariant to monotonic scaling; preserves physical units. Physical sensor attribute. |
| `point_count` | Numerical (float) | Pass-through (no scaling) | Tree-based models (XGBoost/HGB) are invariant to monotonic scaling; preserves physical units. Physical sensor attribute. |
| `relative_velocity_mps` | Numerical (float) | Pass-through (no scaling) | Tree-based models (XGBoost/HGB) are invariant to monotonic scaling; preserves physical units. Physical sensor attribute. |
| `truck_speed_kmph` | Numerical (float) | Pass-through (no scaling) | Tree-based models (XGBoost/HGB) are invariant to monotonic scaling; preserves physical units. Physical sensor attribute. |
| `time_to_collision_s` | Numerical (float) | Pass-through (no scaling) | Tree-based models (XGBoost/HGB) are invariant to monotonic scaling; preserves physical units. Derived kinematic metric. |
| `object_type` | Categorical (str) | One-Hot Encoding (`handle_unknown='ignore'`) | Avoids introducing artificial ordinal hierarchy between distinct vehicle/obstacle categories. Unknown categories handled safely. |

## 3. Target Label Encoding

Categorical target strings were mapped to deterministic integers for multi-class gradient boosting:
```text
SAFE     → 0
CAUTION  → 1
WARNING  → 2
CRITICAL → 3
```
Saved as `models/label_mapping.json` for deterministic reverse-lookup during inference.

## 4. Train / Validation / Test Stratified Split

* **Random Seed**: `42`
* **Partition Ratios**: 70% Train (14,000) / 15% Validation (3,000) / 15% Test (3,000)

| Split Partition | Total Rows | SAFE (%) | CAUTION (%) | WARNING (%) | CRITICAL (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Full Dataset | 20,000 | 44.5% | 23.7% | 16.0% | 15.8% |
| Training (70%) | 14,000 | 44.5% | 23.7% | 16.0% | 15.8% |
| Validation (15%) | 3,000 | 44.5% | 23.7% | 16.0% | 15.8% |
| Testing (15%) | 3,000 | 44.5% | 23.8% | 16.0% | 15.8% |

*Exact stratification achieved: Class proportions vary by less than 0.1% across all three partitions.*

## 5. Processed Feature Dimensions

* **Original Raw Features**: `12`
* **Numerical Features (passed through)**: `11`
* **One-Hot Encoded Categories**: `6` (`car`, `truck`, `crane`, `excavator`, `person`, `unknown`)
* **Final Transformed Feature Count**: `17`

### Transformed Feature Names (Exact Order):
1. `distance_m`
2. `object_x_m`
3. `object_y_m`
4. `object_z_m`
5. `object_width_m`
6. `object_height_m`
7. `object_length_m`
8. `point_count`
9. `relative_velocity_mps`
10. `truck_speed_kmph`
11. `time_to_collision_s`
12. `object_type_car`
13. `object_type_truck`
14. `object_type_crane`
15. `object_type_excavator`
16. `object_type_person`
17. `object_type_unknown`


## 6. Generated Output Files & Artifacts

| Category | File Path | Format | Description |
| :--- | :--- | :--- | :--- |
| **Processed Data** | `data/processed/X_train.csv` | CSV | Transformed training features (14,000 rows × 17 cols) |
| | `data/processed/y_train.csv` | CSV | Encoded training targets (14,000 rows × 1 col) |
| | `data/processed/X_val.csv` | CSV | Transformed validation features (3,000 rows × 17 cols) |
| | `data/processed/y_val.csv` | CSV | Encoded validation targets (3,000 rows × 1 col) |
| | `data/processed/X_test.csv` | CSV | Transformed test features (3,000 rows × 17 cols) |
| | `data/processed/y_test.csv` | CSV | Encoded test targets (3,000 rows × 1 col) |
| **Raw Partitions** | `data/processed/train_raw.csv` | CSV | Untransformed training partition for inspection |
| | `data/processed/val_raw.csv` | CSV | Untransformed validation partition for inspection |
| | `data/processed/test_raw.csv` | CSV | Untransformed test partition for inspection |
| **Metadata** | `models/preprocessor.joblib` | Joblib | Fitted `MineRakshakPreprocessor` transformer |
| | `models/label_mapping.json` | JSON | Target string ↔ integer mapping |
| | `models/feature_schema.json` | JSON | Exact feature ordering specification |
| | `results/metrics/split_distribution.csv` | CSV | Stratified split distribution metrics |

## 7. Data Quality & Reproducibility Verification

* **Reproducibility Test**: `PASSED (100% Bitwise Identical)`
* **Missing / Null Values in Transformed Sets**: `0`
* **Infinite Values in Transformed Sets**: `0`
* **Column Ordering Consistency**: Verified 100% identical across `X_train`, `X_val`, and `X_test`.

## 8. Conclusion & Readiness for Stage 5

The dataset is fully prepared for model training. The preprocessor artifact `models/preprocessor.joblib` will ensure that any future single-item or batch input from the ROS 2 adapter is transformed in exactly the same way before inference.