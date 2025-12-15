from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np

from .features import DEFAULT_FEATURE_SPEC, FeatureSpec, extract_features_from_config, vectorize_features


@dataclass
class RouterModel:
    """Thin wrapper around a trained sklearn pipeline."""

    pipeline: Any
    feature_names: List[str]
    allowed_keys: List[str]
    sa_threshold: Optional[float] = None

    @classmethod
    def load(cls, path: str | Path) -> "RouterModel":
        payload = joblib.load(str(path))
        if isinstance(payload, dict) and "pipeline" in payload:
            pipeline = payload["pipeline"]
            feature_names = list(payload.get("feature_names", DEFAULT_FEATURE_SPEC.names))
            allowed_keys = list(payload.get("allowed_keys", ["sa", "ga", "aco"]))
            sa_threshold = payload.get("sa_threshold")
            sa_threshold = float(sa_threshold) if sa_threshold is not None else None
            return cls(pipeline=pipeline, feature_names=feature_names, allowed_keys=allowed_keys, sa_threshold=sa_threshold)
        # Backward/alternate: payload itself is a pipeline.
        return cls(pipeline=payload, feature_names=DEFAULT_FEATURE_SPEC.names, allowed_keys=["sa", "ga", "aco"], sa_threshold=None)

    def predict_optimizer_key(self, feature_vector: np.ndarray) -> str:
        # If a tuned SA threshold is provided, apply it using probabilities.
        if self.sa_threshold is not None and hasattr(self.pipeline, "predict_proba"):
            proba = self.pipeline.predict_proba(feature_vector.reshape(1, -1))[0]

            classes = None
            if hasattr(self.pipeline, "named_steps") and "clf" in self.pipeline.named_steps and hasattr(
                self.pipeline.named_steps["clf"], "classes_"
            ):
                classes = list(self.pipeline.named_steps["clf"].classes_)
            elif hasattr(self.pipeline, "classes_"):
                classes = list(self.pipeline.classes_)  # type: ignore[attr-defined]

            if classes and "sa" in classes:
                sa_idx = classes.index("sa")
                if float(proba[sa_idx]) >= float(self.sa_threshold):
                    return "sa"
                # Choose best non-SA class.
                best_other = max(
                    [(c, float(proba[classes.index(c)])) for c in classes if c != "sa"],
                    key=lambda kv: kv[1],
                )[0]
                return str(best_other)

        # Default: sklearn predicts labels as provided during training (we train with string keys)
        pred = self.pipeline.predict(feature_vector.reshape(1, -1))[0]
        return str(pred)

    def predict_optimizer_key_from_config(self) -> str:
        spec = FeatureSpec(names=self.feature_names)
        feature_dict = extract_features_from_config(spec)
        vec = vectorize_features(feature_dict, spec)
        return self.predict_optimizer_key(vec)
