"""Stage 8: Model Serialization & Packaging Verification Test Suite.

Ensures that all serialized artifacts:
1. Primary Model: models/hgb_model.joblib
2. Secondary Model: models/xgboost_model.json
3. Preprocessor: models/preprocessor.joblib
4. Feature Schema: models/feature_schema.json
5. Label Mapping: models/label_mapping.json
6. Pipeline Manifest: models/manifest.json / models/model_manifest.json

can be loaded cleanly, verify against SHA-256 hashes, and reproduce the EXACT
predictions and probability distributions obtained during Stage 7 evaluation.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import types
import unittest
from pathlib import Path

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Decoupled loading for sklearn HistGradientBoostingClassifier on Python 3.14 (Windows SAC safe)
import sklearn
if "sklearn.ensemble" not in sys.modules:
    _pkg = types.ModuleType("sklearn.ensemble")
    _pkg.__path__ = [os.path.join(os.path.dirname(sklearn.__file__), "ensemble")]
    sys.modules["sklearn.ensemble"] = _pkg

from sklearn.ensemble._hist_gradient_boosting.gradient_boosting import (
    HistGradientBoostingClassifier,
)

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from src.preprocessing.preprocess import MineRakshakPreprocessor


class TestModelSerializationAndReproducibility(unittest.TestCase):
    """Test suite for serialized model packaging, checksum verification, and prediction reproducibility."""

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize paths and load reference manifests and datasets."""
        cls.models_dir = PROJECT_ROOT / "models"
        cls.metrics_dir = PROJECT_ROOT / "results" / "metrics"
        cls.data_dir = PROJECT_ROOT / "data" / "processed"

        cls.manifest_path = cls.models_dir / "manifest.json"
        cls.model_manifest_path = cls.models_dir / "model_manifest.json"
        cls.metadata_path = cls.models_dir / "model_metadata.json"
        cls.hgb_path = cls.models_dir / "hgb_model.joblib"
        cls.xgb_path = cls.models_dir / "xgboost_model.json"
        cls.prep_path = cls.models_dir / "preprocessor.joblib"
        cls.schema_path = cls.models_dir / "feature_schema.json"
        cls.labels_path = cls.models_dir / "label_mapping.json"

        cls.x_test_path = cls.data_dir / "X_test.csv"
        cls.y_test_path = cls.data_dir / "y_test.csv"
        cls.hgb_saved_preds_path = cls.metrics_dir / "hgb_test_predictions.csv"
        cls.xgb_saved_preds_path = cls.metrics_dir / "xgboost_test_predictions.csv"

        # Load manifest
        with open(cls.manifest_path, "r", encoding="utf-8") as f:
            cls.manifest = json.load(f)

    @staticmethod
    def _compute_sha256(filepath: Path) -> str:
        """Compute SHA-256 hash of a file."""
        hasher = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def test_01_all_required_artifacts_exist(self) -> None:
        """Verify that all 8 required Stage 8 files exist and are non-empty."""
        artifacts = [
            self.manifest_path,
            self.model_manifest_path,
            self.metadata_path,
            self.hgb_path,
            self.xgb_path,
            self.prep_path,
            self.schema_path,
            self.labels_path,
        ]
        for path in artifacts:
            self.assertTrue(path.exists(), f"Artifact missing: {path}")
            self.assertGreater(path.stat().st_size, 0, f"Artifact empty: {path}")

    def test_02_manifest_checksum_integrity(self) -> None:
        """Verify that SHA-256 checksums of all serialized artifacts match the manifest exactly."""
        art = self.manifest["artifacts"]

        expected_hashes = {
            "hgb_model.joblib": art["primary_model"]["sha256"],
            "xgboost_model.json": art["secondary_model"]["sha256"],
            "preprocessor.joblib": art["preprocessor"]["sha256"],
            "feature_schema.json": art["feature_schema"]["sha256"],
            "label_mapping.json": art["label_mapping"]["sha256"],
        }

        for filename, expected_hash in expected_hashes.items():
            actual_hash = self._compute_sha256(self.models_dir / filename)
            self.assertEqual(
                actual_hash,
                expected_hash,
                f"Checksum mismatch for {filename}! Expected: {expected_hash}, Actual: {actual_hash}",
            )

    def test_03_feature_schema_and_labels_consistency(self) -> None:
        """Verify consistency between feature schema, label mapping, and expected configurations."""
        with open(self.schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
        with open(self.labels_path, "r", encoding="utf-8") as f:
            labels = json.load(f)

        self.assertEqual(schema["n_raw_features"], 12)
        self.assertEqual(schema["n_transformed_features"], 17)
        self.assertEqual(len(schema["transformed_feature_names"]), 17)

        expected_classes = ["SAFE", "CAUTION", "WARNING", "CRITICAL"]
        self.assertEqual(list(labels["class_to_int"].keys()), expected_classes)
        self.assertEqual(list(labels["class_to_int"].values()), [0, 1, 2, 3])

    def test_04_preprocessor_loading_and_transformation(self) -> None:
        """Verify that preprocessor.joblib loads cleanly and executes deterministic transformation."""
        preprocessor = joblib.load(self.prep_path)
        self.assertIsInstance(preprocessor, MineRakshakPreprocessor)
        self.assertTrue(preprocessor.is_fitted)

        # Create raw test sample
        raw_sample = pd.DataFrame([{
            "distance_m": 14.5,
            "object_x_m": 2.1,
            "object_y_m": 14.3,
            "object_z_m": 0.4,
            "object_width_m": 2.5,
            "object_height_m": 2.8,
            "object_length_m": 5.0,
            "point_count": 350,
            "relative_velocity_mps": -5.2,
            "truck_speed_kmph": 28.0,
            "time_to_collision_s": 2.78,
            "object_type": "truck",
        }])

        transformed = preprocessor.transform(raw_sample)
        self.assertEqual(transformed.shape, (1, 17))
        self.assertEqual(list(transformed.columns), preprocessor.transformed_feature_names)

        # Test unknown object_type handling (fault tolerance)
        unknown_sample = raw_sample.copy()
        unknown_sample["object_type"] = "unseen_mining_robot"
        transformed_unknown = preprocessor.transform(unknown_sample)
        self.assertEqual(transformed_unknown.shape, (1, 17))
        # Ensure all one-hot columns are 0 without crashing
        one_hot_cols = [c for c in transformed_unknown.columns if c.startswith("object_type_")]
        self.assertEqual(transformed_unknown[one_hot_cols].sum().sum(), 0.0)

    def test_05_primary_hgb_model_reproducibility(self) -> None:
        """Verify that hgb_model.joblib reproduces the EXACT Stage 7 test set predictions and probabilities."""
        self.assertTrue(self.x_test_path.exists())
        self.assertTrue(self.hgb_saved_preds_path.exists())

        X_test = pd.read_csv(self.x_test_path)
        saved_df = pd.read_csv(self.hgb_saved_preds_path)

        model = joblib.load(self.hgb_path)
        self.assertIsInstance(model, HistGradientBoostingClassifier)

        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)

        labels = self.manifest["artifacts"]["label_mapping"]["int_to_class"]
        pred_labels = [labels[str(p)] for p in preds]

        # 1. 100% agreement on predicted risk tier labels
        saved_labels = saved_df["predicted_risk_level"].tolist()
        self.assertEqual(pred_labels, saved_labels, "HGB predictions do not match Stage 7 saved predictions!")

        # 2. Probability matching within float tolerance (atol=1e-5)
        saved_probs = saved_df[
            ["probability_SAFE", "probability_CAUTION", "probability_WARNING", "probability_CRITICAL"]
        ].values
        np.testing.assert_allclose(
            probs,
            saved_probs,
            atol=1e-5,
            err_msg="HGB predicted probabilities deviate from Stage 7 saved outputs!",
        )

    def test_06_secondary_xgboost_model_reproducibility(self) -> None:
        """Verify that xgboost_model.json reproduces the EXACT Stage 7 test set predictions and probabilities."""
        self.assertTrue(self.x_test_path.exists())
        self.assertTrue(self.xgb_saved_preds_path.exists())

        X_test = pd.read_csv(self.x_test_path)
        saved_df = pd.read_csv(self.xgb_saved_preds_path)

        model = xgb.XGBClassifier()
        model.load_model(str(self.xgb_path))

        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)

        labels = self.manifest["artifacts"]["label_mapping"]["int_to_class"]
        pred_labels = [labels[str(p)] for p in preds]

        # 1. 100% agreement on predicted risk tier labels
        saved_labels = saved_df["predicted_risk_level"].tolist()
        self.assertEqual(pred_labels, saved_labels, "XGBoost predictions do not match Stage 7 saved predictions!")

        # 2. Probability matching within float tolerance (atol=1e-5)
        saved_probs = saved_df[
            ["probability_SAFE", "probability_CAUTION", "probability_WARNING", "probability_CRITICAL"]
        ].values
        np.testing.assert_allclose(
            probs,
            saved_probs,
            atol=1e-5,
            err_msg="XGBoost predicted probabilities deviate from Stage 7 saved outputs!",
        )

    def test_07_end_to_end_raw_input_inference(self) -> None:
        """Test full end-to-end inference flow: raw input dictionary -> preprocessor -> HGB & XGBoost models."""
        preprocessor = joblib.load(self.prep_path)
        hgb_model = joblib.load(self.hgb_path)
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model(str(self.xgb_path))

        int_to_class = self.manifest["artifacts"]["label_mapping"]["int_to_class"]

        # Critical scenario: close obstacle, fast approach, low TTC
        raw_critical_sample = pd.DataFrame([{
            "distance_m": 6.2,
            "object_x_m": 0.5,
            "object_y_m": 6.1,
            "object_z_m": 0.2,
            "object_width_m": 3.0,
            "object_height_m": 3.2,
            "object_length_m": 7.0,
            "point_count": 520,
            "relative_velocity_mps": -8.5,
            "truck_speed_kmph": 32.0,
            "time_to_collision_s": 0.73,
            "object_type": "truck",
        }])

        X_transformed = preprocessor.transform(raw_critical_sample)

        # HGB inference
        hgb_pred = int(hgb_model.predict(X_transformed)[0])
        hgb_prob = hgb_model.predict_proba(X_transformed)[0]
        hgb_label = int_to_class[str(hgb_pred)]

        # XGBoost inference
        xgb_pred = int(xgb_model.predict(X_transformed)[0])
        xgb_prob = xgb_model.predict_proba(X_transformed)[0]
        xgb_label = int_to_class[str(xgb_pred)]

        # Assert valid classes and probabilities
        self.assertIn(hgb_pred, [0, 1, 2, 3])
        self.assertIn(xgb_pred, [0, 1, 2, 3])
        self.assertAlmostEqual(float(np.sum(hgb_prob)), 1.0, places=4)
        self.assertAlmostEqual(float(np.sum(xgb_prob)), 1.0, places=4)

        # In this critical scenario, both models should detect severe risk (CRITICAL or WARNING)
        self.assertIn(hgb_label, ["CRITICAL", "WARNING"])
        self.assertIn(xgb_label, ["CRITICAL", "WARNING"])

    def test_08_inference_latency_benchmark(self) -> None:
        """Benchmark single-sample inference latency to verify sub-50ms real-time budget."""
        preprocessor = joblib.load(self.prep_path)
        hgb_model = joblib.load(self.hgb_path)
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model(str(self.xgb_path))

        raw_sample = pd.DataFrame([{
            "distance_m": 22.0,
            "object_x_m": -1.5,
            "object_y_m": 21.9,
            "object_z_m": 0.0,
            "object_width_m": 1.8,
            "object_height_m": 1.6,
            "object_length_m": 4.5,
            "point_count": 210,
            "relative_velocity_mps": -2.0,
            "truck_speed_kmph": 25.0,
            "time_to_collision_s": 11.0,
            "object_type": "car",
        }])

        X_transformed = preprocessor.transform(raw_sample)

        # Warm up
        hgb_model.predict(X_transformed)
        xgb_model.predict(X_transformed)

        # Benchmark HGB
        times_hgb = []
        for _ in range(50):
            t0 = time.perf_counter()
            hgb_model.predict(X_transformed)
            times_hgb.append(time.perf_counter() - t0)
        mean_hgb_ms = float(np.mean(times_hgb) * 1000)

        # Benchmark XGBoost
        times_xgb = []
        for _ in range(50):
            t0 = time.perf_counter()
            xgb_model.predict(X_transformed)
            times_xgb.append(time.perf_counter() - t0)
        mean_xgb_ms = float(np.mean(times_xgb) * 1000)

        # Both must comfortably satisfy haul truck real-time safety budget (< 50ms)
        self.assertLess(mean_hgb_ms, 50.0, f"HGB latency too high: {mean_hgb_ms:.2f} ms")
        self.assertLess(mean_xgb_ms, 50.0, f"XGBoost latency too high: {mean_xgb_ms:.2f} ms")


if __name__ == "__main__":
    unittest.main(verbosity=2)
