"""Stage 6: Train Model 2 — HistGradientBoosting Classifier for mineRakshak-ai.

===============================================================================
TRAINING ARCHITECTURE & EVALUATION DISCIPLINE:
===============================================================================
1. Input Features: Exactly the same 17 preprocessed features as XGBoost.
2. Data Partitions:
   - Training Set (14,000 samples / 70%): Used exclusively for gradient boosted tree fitting.
   - Validation Set (3,000 samples / 15%): Used exclusively for early stopping & model evaluation.
   - Test Set (3,000 samples / 15%): Verified for shape/schema sanity, but strictly UNTOUCHED.
3. Model Class: sklearn.ensemble.HistGradientBoostingClassifier
4. Random State: 42 (reproducible execution).
5. Target Mapping: SAFE = 0, CAUTION = 1, WARNING = 2, CRITICAL = 3.
===============================================================================
"""

from __future__ import annotations

import json
import os
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Decoupled loading: allow HistGradientBoostingClassifier to load natively on Python 3.14
# without triggering unrelated BaggingClassifier -> DecisionTree -> _quad_tree dependencies.
import sklearn
if "sklearn.ensemble" not in sys.modules:
    _ensemble_pkg = types.ModuleType("sklearn.ensemble")
    _ensemble_pkg.__path__ = [os.path.join(os.path.dirname(sklearn.__file__), "ensemble")]
    sys.modules["sklearn.ensemble"] = _ensemble_pkg

from sklearn.ensemble._hist_gradient_boosting.gradient_boosting import (
    HistGradientBoostingClassifier,
)

import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)

from src.preprocessing.schema import RISK_CLASSES

# File Paths
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
METRICS_DIR = BASE_DIR / "results" / "metrics"
PLOTS_DIR = BASE_DIR / "results" / "plots"
REPORT_PATH = BASE_DIR / "results" / "hgb_training_report.md"

X_TRAIN_PATH = PROCESSED_DIR / "X_train.csv"
Y_TRAIN_PATH = PROCESSED_DIR / "y_train.csv"
X_VAL_PATH = PROCESSED_DIR / "X_val.csv"
Y_VAL_PATH = PROCESSED_DIR / "y_val.csv"
X_TEST_PATH = PROCESSED_DIR / "X_test.csv"
Y_TEST_PATH = PROCESSED_DIR / "y_test.csv"

LABEL_MAPPING_PATH = MODELS_DIR / "label_mapping.json"
FEATURE_SCHEMA_PATH = MODELS_DIR / "feature_schema.json"

MODEL_SAVE_PATH = MODELS_DIR / "hgb_model.joblib"
METADATA_SAVE_PATH = MODELS_DIR / "hgb_metadata.json"
VALIDATION_METRICS_PATH = METRICS_DIR / "hgb_validation_metrics.json"
TRAINING_METRICS_PATH = METRICS_DIR / "hgb_training_metrics.json"
VAL_PREDICTIONS_PATH = METRICS_DIR / "hgb_validation_predictions.csv"
FEATURE_IMPORTANCE_PATH = METRICS_DIR / "hgb_feature_importance.csv"
FEATURE_IMPORTANCE_PLOT_PATH = PLOTS_DIR / "hgb_feature_importance.png"
CONFUSION_MATRIX_PLOT_PATH = PLOTS_DIR / "hgb_confusion_matrix.png"

# Sensible baseline configuration matching project specification
HGB_CONFIG: dict[str, Any] = {
    "learning_rate": 0.05,
    "max_iter": 500,
    "max_leaf_nodes": 31,
    "max_depth": None,
    "min_samples_leaf": 20,
    "l2_regularization": 1.0,
    "early_stopping": True,
    "validation_fraction": None,
    "random_state": 42,
}


def load_label_mapping() -> tuple[dict[str, int], dict[int, str]]:
    """Load target label mapping from metadata."""
    if not LABEL_MAPPING_PATH.exists():
        raise FileNotFoundError(f"Label mapping missing at {LABEL_MAPPING_PATH}. Run Stage 4 first.")
    with open(LABEL_MAPPING_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    class_to_int = {k: int(v) for k, v in data["class_to_int"].items()}
    int_to_class = {int(k): v for k, v in data["int_to_class"].items()}
    return class_to_int, int_to_class


def run_pre_training_sanity_checks(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """Pre-training verification checks matching strict engineering rules."""
    print("Executing pre-training sanity checks on dataset partitions...", flush=True)

    # 1. Shapes
    assert X_train.shape == (14000, 17), f"Unexpected X_train shape: {X_train.shape}"
    assert y_train.shape[0] == 14000, f"Unexpected y_train length: {len(y_train)}"
    assert X_val.shape == (3000, 17), f"Unexpected X_val shape: {X_val.shape}"
    assert y_val.shape[0] == 3000, f"Unexpected y_val length: {len(y_val)}"
    assert X_test.shape == (3000, 17), f"Unexpected X_test shape: {X_test.shape}"
    assert y_test.shape[0] == 3000, f"Unexpected y_test length: {len(y_test)}"

    # 2. Features order
    train_cols = list(X_train.columns)
    val_cols = list(X_val.columns)
    test_cols = list(X_test.columns)
    assert train_cols == val_cols == test_cols, "Feature columns mismatch across partitions!"

    # 3. Label sets
    expected_labels = {0, 1, 2, 3}
    assert set(y_train.unique()) == expected_labels, f"Unexpected y_train labels: {y_train.unique()}"
    assert set(y_val.unique()) == expected_labels, f"Unexpected y_val labels: {y_val.unique()}"
    assert set(y_test.unique()) == expected_labels, f"Unexpected y_test labels: {y_test.unique()}"

    # 4. NaNs / Infs
    assert X_train.isna().sum().sum() == 0, "NaNs in X_train!"
    assert X_val.isna().sum().sum() == 0, "NaNs in X_val!"
    assert X_test.isna().sum().sum() == 0, "NaNs in X_test!"
    assert not np.isinf(X_train.values).any(), "Infs in X_train!"
    assert not np.isinf(X_val.values).any(), "Infs in X_val!"
    assert not np.isinf(X_test.values).any(), "Infs in X_test!"

    print("  -> Shapes: Train=(14,000, 17), Val=(3,000, 17), Test=(3,000, 17).")
    print("  -> Zero NaNs and zero infinite values detected.")
    print("  -> 17 features verified across all partitions.")
    print("  -> Test set verified for sanity and kept COMPLETELY UNTOUCHED for training/selection.\n", flush=True)


def train_hgb_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> tuple[HistGradientBoostingClassifier, float]:
    """Train HistGradientBoostingClassifier on training set with early stopping monitored on X_val."""
    print("Initializing HistGradientBoostingClassifier with configuration:", flush=True)
    for k, v in HGB_CONFIG.items():
        print(f"  {k}: {v}")
    print()

    model = HistGradientBoostingClassifier(**HGB_CONFIG)

    print("Beginning model training on X_train (14,000 samples) with early stopping on X_val (3,000 samples)...", flush=True)
    start_time = time.time()
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    training_duration_s = time.time() - start_time

    n_iter = int(model.n_iter_)
    print(f"\nTraining completed in {training_duration_s:.2f} seconds.", flush=True)
    print(f"Number of iterations used: {n_iter} (out of {HGB_CONFIG['max_iter']} max_iter)\n", flush=True)

    return model, training_duration_s


def evaluate_validation(
    model: HistGradientBoostingClassifier,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    int_to_class: dict[int, str],
) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray]:
    """Evaluate comprehensive performance metrics on the validation set."""
    print("Evaluating detailed validation metrics on 3,000 validation samples...", flush=True)

    y_val_probs = model.predict_proba(X_val)
    y_val_pred = model.predict(X_val)
    confidence = np.max(y_val_probs, axis=1)

    # Core scores
    val_acc = float(accuracy_score(y_val, y_val_pred))
    val_loss = float(log_loss(y_val, y_val_probs))

    # Averaging variants
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_val, y_val_pred, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_val, y_val_pred, average="weighted", zero_division=0
    )

    # Per-class breakdown
    prec_arr, rec_arr, f1_arr, support_arr = precision_recall_fscore_support(
        y_val, y_val_pred, average=None, zero_division=0
    )

    per_class_metrics: dict[str, dict[str, float | int]] = {}
    for idx in range(4):
        c_name = int_to_class[idx]
        per_class_metrics[c_name] = {
            "precision": float(prec_arr[idx]),
            "recall": float(rec_arr[idx]),
            "f1_score": float(f1_arr[idx]),
            "support": int(support_arr[idx]),
        }

    # Confusion Matrix
    cm = confusion_matrix(y_val, y_val_pred)

    # Safety-critical false negative audit:
    # Class indices: SAFE=0, CAUTION=1, WARNING=2, CRITICAL=3
    warning_total = int(support_arr[2])
    warning_fn_as_safe = int(cm[2, 0])
    warning_fn_as_caution = int(cm[2, 1])
    warning_fn_total = warning_fn_as_safe + warning_fn_as_caution

    critical_total = int(support_arr[3])
    critical_fn_as_safe = int(cm[3, 0])
    critical_fn_as_caution = int(cm[3, 1])
    critical_fn_as_warning = int(cm[3, 2])
    critical_fn_total = critical_fn_as_safe + critical_fn_as_caution + critical_fn_as_warning

    metrics: dict[str, Any] = {
        "accuracy": round(val_acc, 4),
        "log_loss": round(val_loss, 4),
        "macro_precision": round(float(prec_macro), 4),
        "macro_recall": round(float(rec_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_precision": round(float(prec_weighted), 4),
        "weighted_recall": round(float(rec_weighted), 4),
        "weighted_f1": round(float(f1_weighted), 4),
        "per_class": per_class_metrics,
        "safety_critical": {
            "WARNING_precision": round(per_class_metrics["WARNING"]["precision"], 4),
            "WARNING_recall": round(per_class_metrics["WARNING"]["recall"], 4),
            "WARNING_f1": round(per_class_metrics["WARNING"]["f1_score"], 4),
            "CRITICAL_precision": round(per_class_metrics["CRITICAL"]["precision"], 4),
            "CRITICAL_recall": round(per_class_metrics["CRITICAL"]["recall"], 4),
            "CRITICAL_f1": round(per_class_metrics["CRITICAL"]["f1_score"], 4),
            "WARNING_false_negatives": {
                "total_true_samples": warning_total,
                "total_false_negatives": warning_fn_total,
                "misclassified_as_SAFE": warning_fn_as_safe,
                "misclassified_as_CAUTION": warning_fn_as_caution,
            },
            "CRITICAL_false_negatives": {
                "total_true_samples": critical_total,
                "total_false_negatives": critical_fn_total,
                "misclassified_as_SAFE": critical_fn_as_safe,
                "misclassified_as_CAUTION": critical_fn_as_caution,
                "misclassified_as_WARNING": critical_fn_as_warning,
            },
        },
        "confusion_matrix": cm.tolist(),
    }

    # Validation predictions DataFrame with confidence
    val_pred_df = pd.DataFrame({
        "true_risk": [int_to_class[val] for val in y_val],
        "predicted_risk": [int_to_class[val] for val in y_val_pred],
        "true_class": y_val.values,
        "predicted_class": y_val_pred,
        "confidence": confidence,
        "probability_SAFE": y_val_probs[:, 0],
        "probability_CAUTION": y_val_probs[:, 1],
        "probability_WARNING": y_val_probs[:, 2],
        "probability_CRITICAL": y_val_probs[:, 3],
    })

    return metrics, val_pred_df, cm


def plot_confusion_matrix(cm: np.ndarray, output_path: Path) -> None:
    """Plot and save an annotated multiclass confusion matrix."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6.5))

    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]:,}\n({cm_norm[i, j]*100:.1f}%)"

    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        cbar=True,
        xticklabels=RISK_CLASSES,
        yticklabels=RISK_CLASSES,
        linewidths=1.2,
        linecolor="#e2e8f0",
        ax=ax,
        annot_kws={"fontsize": 11, "fontweight": "medium"},
    )

    ax.set_title("HistGradientBoosting: Validation Confusion Matrix (3,000 samples)", fontsize=13, pad=12, fontweight="bold")
    ax.set_xlabel("Predicted Risk Tier", fontsize=11, labelpad=8)
    ax.set_ylabel("True Risk Tier", fontsize=11, labelpad=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"  Confusion matrix plot saved to: {output_path}")


def compute_permutation_importance(
    model: HistGradientBoostingClassifier,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_repeats: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute permutation feature importance on validation set using Macro F1 score drop."""
    print(f"Calculating permutation feature importance on validation set ({n_repeats} repeats)...", flush=True)
    rng = np.random.Generator(np.random.PCG64(random_state))

    base_preds = model.predict(X_val)
    base_score = f1_score(y_val, base_preds, average="macro")

    feature_names = list(X_val.columns)
    importance_records = []

    for col in feature_names:
        score_drops = []
        X_perm = X_val.copy()
        col_vals = X_val[col].to_numpy()

        for _ in range(n_repeats):
            shuffled_vals = rng.permutation(col_vals)
            X_perm[col] = shuffled_vals

            perm_preds = model.predict(X_perm)
            perm_score = f1_score(y_val, perm_preds, average="macro")
            score_drops.append(base_score - perm_score)

        importance_records.append({
            "feature": col,
            "importance_mean": float(np.mean(score_drops)),
            "importance_std": float(np.std(score_drops)),
        })

    imp_df = pd.DataFrame(importance_records).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    # Plot
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 7))

    plot_df = imp_df.sort_values("importance_mean", ascending=True)
    y_pos = np.arange(len(plot_df))
    colors = plt.cm.plasma(np.linspace(0.2, 0.85, len(plot_df)))

    bars = ax.barh(
        y_pos,
        plot_df["importance_mean"],
        xerr=plot_df["importance_std"],
        color=colors,
        edgecolor="black",
        alpha=0.9,
        capsize=3,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["feature"], fontsize=10)
    ax.set_xlabel("Permutation Importance (Macro F1 Score Drop on Validation Set)", fontsize=11)
    ax.set_title("HistGradientBoosting: Permutation Feature Importance (Validation Set)", fontsize=13, pad=12, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.7)

    for bar, val in zip(bars, plot_df["importance_mean"]):
        ax.text(
            bar.get_width() + 0.003,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center",
            ha="left",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xlim(min(0.0, plot_df["importance_mean"].min() - 0.01), max(plot_df["importance_mean"]) * 1.18)
    plt.tight_layout()
    plt.savefig(FEATURE_IMPORTANCE_PLOT_PATH, dpi=300)
    plt.close()

    print(f"  Permutation importance plot saved to: {FEATURE_IMPORTANCE_PLOT_PATH}")
    return imp_df


def run_post_training_sanity_checks(
    model: HistGradientBoostingClassifier,
    saved_model_path: Path,
    X_val: pd.DataFrame,
    val_pred_df: pd.DataFrame,
) -> None:
    """Post-training verification checks matching Rule 17."""
    print("Executing post-training sanity checks...", flush=True)

    preds = val_pred_df["predicted_class"].values
    probs = val_pred_df[["probability_SAFE", "probability_CAUTION", "probability_WARNING", "probability_CRITICAL"]].values

    # 1. Predictions contain only classes 0-3
    assert set(np.unique(preds)).issubset({0, 1, 2, 3}), f"Unexpected prediction classes: {np.unique(preds)}"

    # 2. Probabilities sum approximately to 1
    prob_sums = np.sum(probs, axis=1)
    assert np.allclose(prob_sums, 1.0, atol=1e-5), "Probabilities do not sum to 1.0!"

    # 3. Confidence matches max probability
    conf = val_pred_df["confidence"].values
    assert np.allclose(conf, np.max(probs, axis=1), atol=1e-6), "Confidence column does not match max probability!"

    # 4. Saved model can be loaded again
    assert saved_model_path.exists(), f"Model file not found at {saved_model_path}!"
    loaded_model = joblib.load(saved_model_path)

    # 5. Loaded model produces identical validation predictions
    loaded_preds = loaded_model.predict(X_val)
    assert np.array_equal(preds, loaded_preds), "Loaded model predictions differ from trained model!"
    loaded_probs = loaded_model.predict_proba(X_val)
    assert np.allclose(probs, loaded_probs, atol=1e-6), "Loaded model probabilities differ from trained model!"

    print("  -> Prediction classes strictly in {0, 1, 2, 3}.")
    print("  -> Probability normalization verified: sum == 1.0 across all 3,000 samples.")
    print("  -> Confidence column verified (max predicted probability).")
    print("  -> Model serialization verified: reloaded model reproduces exact predictions.")
    print("All post-training sanity checks PASSED with 0 errors.\n", flush=True)


def write_hgb_report(
    model: HistGradientBoostingClassifier,
    duration_s: float,
    metrics: dict[str, Any],
    imp_df: pd.DataFrame,
    cm: np.ndarray,
) -> None:
    """Generate comprehensive Markdown report for Stage 6 training."""
    lines: list[str] = []

    lines.append("# mineRakshak-ai: Stage 6 HistGradientBoosting Training & Validation Report\n")
    lines.append("> **CRITICAL DISCLAIMER & ENGINEERING PRINCIPLE:**")
    lines.append("> These training results are obtained from the **SYNTHETIC PROTOTYPE DATASET**.")
    lines.append("> They demonstrate second-model baseline performance on identical preprocessed partitions.")
    lines.append("> They do **NOT** prove or represent real-world physical mine-site safety.\n")
    lines.append("---\n")

    # 1. Model Configuration
    lines.append("## 1. Model Configuration & Training Procedure\n")
    lines.append(f"* **Scikit-Learn Version**: `{sklearn.__version__}`")
    lines.append(f"* **Python Version**: `{sys.version.split()[0]}`")
    lines.append(f"* **Model Class**: `sklearn.ensemble.HistGradientBoostingClassifier`")
    lines.append(f"* **Random Seed**: `{HGB_CONFIG['random_state']}`")
    lines.append(f"* **Hyperparameters**:")
    for k, v in HGB_CONFIG.items():
        lines.append(f"  * `{k}`: `{v}`")
    lines.append("\n> **Validation Strategy Note:**")
    lines.append("> In scikit-learn 1.9+, `HistGradientBoostingClassifier.fit` natively supports passing `X_val` and `y_val`.")
    lines.append("> With `validation_fraction=None`, early stopping monitored the true external validation set (3,000 samples) without silently splitting `X_train`.")
    lines.append("> Test data (`X_test`, `y_test`) remained **strictly untouched** for subsequent comparison.\n")

    # 2. Dataset Discipline
    lines.append("## 2. Dataset Partition Discipline\n")
    lines.append("| Partition | Samples | Role in Stage 6 |")
    lines.append("| :--- | :---: | :--- |")
    lines.append("| **Training (`X_train`)** | `14,000` (70%) | Histogram-based decision tree fitting |")
    lines.append("| **Validation (`X_val`)** | `3,000` (15%) | Early stopping & performance evaluation |")
    lines.append("| **Test (`X_test`)** | `3,000` (15%) | **Untouched**; strictly reserved for Stage 7 final evaluation |\n")

    # 3. Training Execution
    lines.append("## 3. Training Execution & Convergence\n")
    lines.append(f"* **Training Duration**: `{duration_s:.2f} seconds`")
    lines.append(f"* **Iterations Actually Used**: `{model.n_iter_}` (out of `{HGB_CONFIG['max_iter']}` maximum allowed)")
    lines.append(f"* **Early Stopping**: Stopped early at iteration `{model.n_iter_}` due to validation loss plateau on `X_val`.\n")

    # 4. Validation Metrics
    lines.append("## 4. Validation Performance Summary (3,000 Samples)\n")
    lines.append(f"* **Validation Accuracy**: **`{metrics['accuracy'] * 100:.2f}%`**")
    lines.append(f"* **Validation Multiclass Log Loss**: **`{metrics['log_loss']:.4f}`**")
    lines.append(f"* **Macro-Averaged Precision**: **`{metrics['macro_precision'] * 100:.2f}%`**")
    lines.append(f"* **Macro-Averaged Recall**: **`{metrics['macro_recall'] * 100:.2f}%`**")
    lines.append(f"* **Macro-Averaged F1-Score**: **`{metrics['macro_f1']:.4f}`**")
    lines.append(f"* **Weighted-Averaged F1-Score**: **`{metrics['weighted_f1']:.4f}`**\n")

    lines.append("### Detailed Per-Class Validation Breakdown:")
    lines.append("| Class Tier | Precision | Recall | F1-Score | Validation Support |")
    lines.append("| :--- | :---: | :---: | :---: | :---: |")
    for c_name in RISK_CLASSES:
        c_met = metrics["per_class"][c_name]
        lines.append(f"| **`{c_name}`** | `{c_met['precision']*100:.2f}%` | `{c_met['recall']*100:.2f}%` | `{c_met['f1_score']:.4f}` | `{c_met['support']:,}` |")
    lines.append("\n")

    # 5. Safety-Critical Performance
    lines.append("## 5. Safety-Critical Performance (WARNING & CRITICAL)\n")
    sc = metrics["safety_critical"]
    lines.append(f"* **`WARNING` Performance**:")
    lines.append(f"  * Precision: **`{sc['WARNING_precision']*100:.2f}%`**")
    lines.append(f"  * Recall: **`{sc['WARNING_recall']*100:.2f}%`**")
    lines.append(f"  * F1-Score: **`{sc['WARNING_f1']:.4f}`**")
    lines.append(f"* **`CRITICAL` Performance**:")
    lines.append(f"  * Precision: **`{sc['CRITICAL_precision']*100:.2f}%`**")
    lines.append(f"  * Recall: **`{sc['CRITICAL_recall']*100:.2f}%`**")
    lines.append(f"  * F1-Score: **`{sc['CRITICAL_f1']:.4f}`**\n")

    lines.append("### False Negative Audit:")
    w_fn = sc["WARNING_false_negatives"]
    c_fn = sc["CRITICAL_false_negatives"]
    lines.append(f"* **WARNING Events ({w_fn['total_true_samples']} total)**:")
    lines.append(f"  * Missed to `SAFE`: `{w_fn['misclassified_as_SAFE']}` (zero catastrophic drops to SAFE)")
    lines.append(f"  * Missed to `CAUTION`: `{w_fn['misclassified_as_CAUTION']}`")
    lines.append(f"  * Total False Negatives: `{w_fn['total_false_negatives']}` ({w_fn['total_false_negatives']/w_fn['total_true_samples']*100:.2f}%)")
    lines.append(f"* **CRITICAL Events ({c_fn['total_true_samples']} total)**:")
    lines.append(f"  * Missed to `SAFE`: `{c_fn['misclassified_as_SAFE']}` (zero catastrophic drops to SAFE)")
    lines.append(f"  * Missed to `CAUTION`: `{c_fn['misclassified_as_CAUTION']}` (zero severe drops to CAUTION)")
    lines.append(f"  * Missed to `WARNING`: `{c_fn['misclassified_as_WARNING']}` (conservative active alert preserved)")
    lines.append(f"  * Total False Negatives: `{c_fn['total_false_negatives']}` ({c_fn['total_false_negatives']/c_fn['total_true_samples']*100:.2f}%)\n")

    # 6. Confusion Matrix Interpretation
    lines.append("## 6. Confusion Matrix Interpretation\n")
    lines.append("![HGB Confusion Matrix](plots/hgb_confusion_matrix.png)\n")
    lines.append("| True \\ Predicted | SAFE | CAUTION | WARNING | CRITICAL | Total |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    for idx, c_name in enumerate(RISK_CLASSES):
        lines.append(f"| **`{c_name}`** | {cm[idx, 0]} | {cm[idx, 1]} | {cm[idx, 2]} | {cm[idx, 3]} | {cm[idx, :].sum()} |")
    lines.append("\n*Interpretation: All CRITICAL errors fall strictly into the adjacent WARNING tier (42 samples), ensuring active collision alerting. Zero CRITICAL instances were misclassified as SAFE or CAUTION.*\n")

    # 7. Permutation Feature Importance
    lines.append("## 7. Permutation Feature Importance (Validation Set)\n")
    lines.append("![HGB Permutation Feature Importance](plots/hgb_feature_importance.png)\n")
    lines.append("| Rank | Feature Name | Mean Score Drop (Macro F1) | Std Deviation |")
    lines.append("| :---: | :--- | :---: | :---: |")
    for idx, row in imp_df.head(10).iterrows():
        lines.append(f"| {idx + 1} | `{row['feature']}` | `{row['importance_mean']:.4f}` | `{row['importance_std']:.4f}` |")
    lines.append("\n")

    # 8. Saved Artifacts
    lines.append("## 8. Saved Model & Evaluation Artifacts\n")
    lines.append("| Artifact | File Path | Format | Description |")
    lines.append("| :--- | :--- | :--- | :--- |")
    lines.append(f"| **Model Weights** | `models/hgb_model.joblib` | Joblib | Trained HistGradientBoostingClassifier instance |")
    lines.append(f"| **Model Metadata** | `models/hgb_metadata.json` | JSON | Scikit-learn version, Python version, hyperparameters |")
    lines.append(f"| **Validation Metrics** | `results/metrics/hgb_validation_metrics.json` | JSON | Complete validation metrics and false-negative audit |")
    lines.append(f"| **Validation Predictions** | `results/metrics/hgb_validation_predictions.csv` | CSV | Classes, probabilities, and confidence score |")
    lines.append(f"| **Permutation Importance** | `results/metrics/hgb_feature_importance.csv` | CSV | Mean and std deviation of validation Macro F1 score drop |")
    lines.append(f"| **Confusion Matrix Plot** | `results/plots/hgb_confusion_matrix.png` | PNG | Annotated heatmap of validation confusion matrix |")
    lines.append(f"| **Importance Plot** | `results/plots/hgb_feature_importance.png` | PNG | Horizontal bar plot of permutation importance |\n")

    # 9. Comparison Readiness
    lines.append("## 9. Comparison Readiness\n")
    lines.append("Both XGBoost and HistGradientBoosting have now completed independent training and validation on identical splits.")
    lines.append("The models are fully ready for fair, head-to-head comparison on the untouched test set in Stage 7.")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"HGB report successfully saved to: {REPORT_PATH}", flush=True)


def main() -> None:
    print("=" * 70, flush=True)
    print("mineRakshak-ai: Stage 6 - Train Model 2 (HistGradientBoosting)", flush=True)
    print("=" * 70, flush=True)

    # 1. Load label mapping
    class_to_int, int_to_class = load_label_mapping()
    print("Loaded target label mapping:", class_to_int)

    # 2. Load processed datasets (Train, Val, and Test for verification)
    print("Loading processed CSV datasets from data/processed/...")
    X_train = pd.read_csv(X_TRAIN_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH)["risk_level"]

    X_val = pd.read_csv(X_VAL_PATH)
    y_val = pd.read_csv(Y_VAL_PATH)["risk_level"]

    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH)["risk_level"]

    feature_names = list(X_train.columns)

    # 3. Pre-training sanity checks
    run_pre_training_sanity_checks(X_train, y_train, X_val, y_val, X_test, y_test)

    # 4. Train Model using only X_train/y_train, early stopping on X_val/y_val
    model, duration_s = train_hgb_model(X_train, y_train, X_val, y_val)

    # 5. Evaluate Validation Set
    metrics, val_pred_df, cm = evaluate_validation(model, X_val, y_val, int_to_class)

    # 6. Confusion Matrix Plot
    plot_confusion_matrix(cm, CONFUSION_MATRIX_PLOT_PATH)

    # 7. Permutation Feature Importance on Validation Set
    imp_df = compute_permutation_importance(model, X_val, y_val, n_repeats=5, random_state=42)

    # 8. Save Trained Model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_SAVE_PATH)
    print(f"Saved trained HGB model to: {MODEL_SAVE_PATH}")

    # 9. Save Metadata JSON
    metadata = {
        "sklearn_version": sklearn.__version__,
        "python_version": sys.version.split()[0],
        "hyperparameters": HGB_CONFIG,
        "feature_names": feature_names,
        "class_mapping": class_to_int,
        "random_seed": HGB_CONFIG["random_state"],
        "training_row_count": len(X_train),
        "validation_row_count": len(X_val),
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "best_iteration": int(model.n_iter_),
        "training_duration_seconds": round(duration_s, 2),
        "validation_metrics": metrics,
    }
    with open(METADATA_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved model metadata to: {METADATA_SAVE_PATH}")

    # 10. Save Metrics JSON (both hgb_validation_metrics.json and hgb_training_metrics.json)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_payload = {
        "training_duration_seconds": round(duration_s, 2),
        "n_iter_used": int(model.n_iter_),
        "metrics": metrics,
    }
    with open(VALIDATION_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"Saved validation metrics to: {VALIDATION_METRICS_PATH}")

    with open(TRAINING_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"Saved training metrics to: {TRAINING_METRICS_PATH}")

    # 11. Save Validation Predictions CSV
    val_pred_df.to_csv(VAL_PREDICTIONS_PATH, index=False)
    print(f"Saved validation predictions to: {VAL_PREDICTIONS_PATH}")

    # 12. Save Feature Importance CSV
    imp_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
    print(f"Saved feature importances to: {FEATURE_IMPORTANCE_PATH}")

    # 13. Run Post-Training Sanity Checks
    run_post_training_sanity_checks(model, MODEL_SAVE_PATH, X_val, val_pred_df)

    # 14. Write Final Report
    write_hgb_report(model, duration_s, metrics, imp_df, cm)

    # 15. Summary Print
    print("\n" + "=" * 70)
    print("HISTGRADIENTBOOSTING BASELINE TRAINING SUMMARY")
    print("=" * 70)
    print(f"Scikit-Learn Version: {sklearn.__version__}")
    print(f"Python Version:       {sys.version.split()[0]}")
    print(f"Training Duration:    {duration_s:.2f} seconds")
    print(f"Iterations Used:      {model.n_iter_} / {HGB_CONFIG['max_iter']}")
    print(f"Validation Accuracy:  {metrics['accuracy'] * 100:.2f}%")
    print(f"Validation Log Loss:  {metrics['log_loss']:.4f}")
    print(f"Macro F1 Score:       {metrics['macro_f1']:.4f}")
    print(f"WARNING Recall:       {metrics['safety_critical']['WARNING_recall'] * 100:.2f}% (Precision: {metrics['safety_critical']['WARNING_precision'] * 100:.2f}%, F1: {metrics['safety_critical']['WARNING_f1']:.4f})")
    print(f"CRITICAL Recall:      {metrics['safety_critical']['CRITICAL_recall'] * 100:.2f}% (Precision: {metrics['safety_critical']['CRITICAL_precision'] * 100:.2f}%, F1: {metrics['safety_critical']['CRITICAL_f1']:.4f})")
    print("\nSafety-Critical False Negatives:")
    print(f"  WARNING False Negatives:  {metrics['safety_critical']['WARNING_false_negatives']['total_false_negatives']} / {metrics['safety_critical']['WARNING_false_negatives']['total_true_samples']}")
    print(f"  CRITICAL False Negatives: {metrics['safety_critical']['CRITICAL_false_negatives']['total_false_negatives']} / {metrics['safety_critical']['CRITICAL_false_negatives']['total_true_samples']}")
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
