from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from typing import List, Tuple

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

import config
from optimization.router.features import DEFAULT_FEATURE_SPEC


def _load_dataset(path: str, feature_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    x_rows: List[List[float]] = []
    y_rows: List[str] = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            x_rows.append([float(row[f"f_{n}"]) for n in feature_names])
            y_rows.append(str(row["label"]))
    return np.array(x_rows, dtype=float), np.array(y_rows, dtype=object)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the SA/GA/ACO router model.")
    parser.add_argument("--data", type=str, default="runs/router/dataset.csv")
    parser.add_argument("--out", type=str, default=getattr(config, "ROUTER_MODEL_PATH", "runs/router/model.joblib"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--model",
        type=str,
        default="rf",
        choices=["rf", "mlp"],
        help="Classifier to train. 'rf' is recommended for highly imbalanced data.",
    )
    parser.add_argument(
        "--rf-estimators",
        type=int,
        default=200,
        help="Number of trees when using --model rf.",
    )
    parser.add_argument(
        "--class-weights",
        action="store_true",
        help="Use balanced per-sample weights during training to mitigate class imbalance (recommended when SA is rare).",
    )
    parser.add_argument(
        "--optimize-sa-threshold",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Tune an SA probability threshold on a validation split to improve SA precision/recall tradeoff.",
    )
    args = parser.parse_args()

    feature_names = DEFAULT_FEATURE_SPEC.names
    X, y = _load_dataset(args.data, feature_names)
    if X.size == 0:
        raise SystemExit(f"Empty dataset: {args.data}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y if len(set(y.tolist())) > 1 else None,
    )

    model_name = str(args.model).lower().strip()
    if model_name == "mlp":
        clf = MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            alpha=1e-4,
            max_iter=300,
            random_state=args.seed,
        )
        pipeline = Pipeline(steps=[("scaler", StandardScaler()), ("clf", clf)])
    else:
        # Trees handle skewed classes better; enable class weights when requested.
        clf = RandomForestClassifier(
            n_estimators=int(args.rf_estimators),
            random_state=args.seed,
            n_jobs=-1,
            min_samples_leaf=2,
            class_weight="balanced_subsample" if bool(args.class_weights) else None,
        )
        pipeline = Pipeline(steps=[("clf", clf)])

    fit_kwargs = {}
    if bool(args.class_weights) and model_name == "mlp":
        # MLPClassifier supports per-sample weighting via sample_weight.
        # Pipeline requires the parameter to be prefixed with the step name.
        fit_kwargs["clf__sample_weight"] = compute_sample_weight(class_weight="balanced", y=y_train)

    sa_threshold = None
    if bool(args.optimize_sa_threshold) and hasattr(pipeline, "predict_proba"):
        # Hold out a validation split to tune an SA probability threshold.
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.2,
            random_state=args.seed + 1,
            stratify=y_train if len(set(y_train.tolist())) > 1 else None,
        )
        pipeline.fit(X_tr, y_tr, **fit_kwargs)
        proba = pipeline.predict_proba(X_val)

        # Determine class order (Pipeline doesn't always expose classes_).
        classes = None
        if hasattr(pipeline, "named_steps") and "clf" in pipeline.named_steps and hasattr(pipeline.named_steps["clf"], "classes_"):
            classes = list(pipeline.named_steps["clf"].classes_)
        elif hasattr(pipeline, "classes_"):
            classes = list(pipeline.classes_)  # type: ignore[attr-defined]

        if classes is not None and "sa" in classes:
            sa_idx = classes.index("sa")
            # Grid-search threshold to maximize F1 for SA (one-vs-rest).
            best_t = 0.5
            best_f1 = -1.0
            y_is_sa = (y_val == "sa")
            for t in np.linspace(0.0, 1.0, 201):
                pred_is_sa = proba[:, sa_idx] >= float(t)
                tp = int(np.sum(pred_is_sa & y_is_sa))
                fp = int(np.sum(pred_is_sa & ~y_is_sa))
                fn = int(np.sum(~pred_is_sa & y_is_sa))
                denom = (2 * tp + fp + fn)
                f1 = (2 * tp / denom) if denom else 0.0
                if f1 > best_f1:
                    best_f1 = f1
                    best_t = float(t)
            sa_threshold = best_t

    # Refit on the full training split for the final model.
    pipeline.fit(X_train, y_train, **fit_kwargs)

    if sa_threshold is not None and hasattr(pipeline, "predict_proba"):
        proba_test = pipeline.predict_proba(X_test)
        if hasattr(pipeline, "named_steps") and "clf" in pipeline.named_steps and hasattr(pipeline.named_steps["clf"], "classes_"):
            classes = list(pipeline.named_steps["clf"].classes_)
        else:
            classes = list(getattr(pipeline, "classes_", ["sa", "ga", "aco"]))
        sa_idx = classes.index("sa") if "sa" in classes else None
        y_pred: List[str] = []
        for p in proba_test:
            if sa_idx is not None and float(p[sa_idx]) >= float(sa_threshold):
                y_pred.append("sa")
            else:
                # Choose the best non-SA class.
                best_other = max(
                    [(c, float(p[classes.index(c)])) for c in classes if c != "sa"],
                    key=lambda kv: kv[1],
                )[0]
                y_pred.append(str(best_other))
        y_pred = np.array(y_pred, dtype=object)
    else:
        y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_test, y_pred, labels=["sa", "ga", "aco"]))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))
    if sa_threshold is not None:
        print(f"\nTuned SA threshold: {sa_threshold:.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    payload = {
        "pipeline": pipeline,
        "feature_names": feature_names,
        "allowed_keys": ["sa", "ga", "aco"],
        "sa_threshold": sa_threshold,
    }
    joblib.dump(payload, args.out)
    print(f"\nSaved model to: {args.out}")


if __name__ == "__main__":
    main()
