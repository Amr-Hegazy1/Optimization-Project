from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LABELS_DEFAULT = ["sa", "ga", "aco"]


@dataclass(frozen=True)
class ModelPayload:
    pipeline: Any
    feature_names: List[str]
    allowed_keys: List[str]
    sa_threshold: Optional[float]


def _load_payload(model_path: Path) -> ModelPayload:
    payload = joblib.load(str(model_path))
    if isinstance(payload, dict) and "pipeline" in payload:
        pipeline = payload["pipeline"]
        feature_names = list(payload.get("feature_names", []))
        allowed_keys = list(payload.get("allowed_keys", LABELS_DEFAULT))
        sa_threshold = payload.get("sa_threshold")
        sa_threshold = float(sa_threshold) if sa_threshold is not None else None
        if not feature_names:
            raise SystemExit("Model payload missing feature_names")
        return ModelPayload(
            pipeline=pipeline,
            feature_names=feature_names,
            allowed_keys=allowed_keys,
            sa_threshold=sa_threshold,
        )

    # Backward/alternate format: payload is the pipeline.
    return ModelPayload(
        pipeline=payload,
        feature_names=[],
        allowed_keys=LABELS_DEFAULT,
        sa_threshold=None,
    )


def _get_estimator(pipeline: Any) -> Any:
    # sklearn Pipeline
    if hasattr(pipeline, "named_steps") and "clf" in pipeline.named_steps:
        return pipeline.named_steps["clf"]
    return pipeline


def _get_classes(pipeline: Any) -> Optional[List[str]]:
    est = _get_estimator(pipeline)
    if hasattr(est, "classes_"):
        return [str(x) for x in list(est.classes_)]
    if hasattr(pipeline, "classes_"):
        return [str(x) for x in list(pipeline.classes_)]
    return None


def predict_with_optional_sa_threshold(pipeline: Any, X: np.ndarray, sa_threshold: Optional[float]) -> np.ndarray:
    if sa_threshold is None or not hasattr(pipeline, "predict_proba"):
        return np.asarray(pipeline.predict(X), dtype=object)

    classes = _get_classes(pipeline)
    if not classes or "sa" not in classes:
        return np.asarray(pipeline.predict(X), dtype=object)

    proba = pipeline.predict_proba(X)
    sa_idx = classes.index("sa")
    non_sa = [i for i, c in enumerate(classes) if c != "sa"]
    best_non_sa_idx = np.asarray(non_sa, dtype=int)[np.argmax(proba[:, non_sa], axis=1)]
    best_non_sa_label = np.asarray([classes[i] for i in best_non_sa_idx], dtype=object)
    return np.where(proba[:, sa_idx] >= float(sa_threshold), "sa", best_non_sa_label).astype(object)


def _load_dataset(data_path: Path, feature_names: List[str]) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    df = pd.read_csv(data_path)
    feat_cols = [f"f_{n}" for n in feature_names]
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"Dataset missing feature columns: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if "label" not in df.columns:
        raise SystemExit("Dataset missing 'label' column")

    X = df[feat_cols].to_numpy(dtype=float)
    y = df["label"].astype(str).to_numpy(dtype=object)
    return X, y, df


def _write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Feature importance analysis for the SA/GA/ACO router.")
    p.add_argument("--model", type=str, default="runs/router/model.joblib")
    p.add_argument("--data", type=str, default="runs/router/dataset_combined_min_cost.csv")
    p.add_argument("--out-dir", type=str, default="runs/router/analysis")
    p.add_argument("--topk", type=int, default=15)
    p.add_argument(
        "--permutation-repeats",
        type=int,
        default=10,
        help="Permutation importance repeats (higher is more stable but slower).",
    )
    p.add_argument(
        "--save-pdfs",
        action="store_true",
        help="If set, save publication-ready PDF figures to <out-dir>/figs.",
    )
    args = p.parse_args()

    model_path = Path(args.model)
    data_path = Path(args.data)
    out_dir = Path(args.out_dir)

    payload = _load_payload(model_path)
    if not payload.feature_names:
        raise SystemExit(
            "Model does not include feature_names. Re-train using the provided router_train.py to embed them."
        )

    X, y, df = _load_dataset(data_path, payload.feature_names)
    y_pred = predict_with_optional_sa_threshold(payload.pipeline, X, payload.sa_threshold)

    label_counts = df["label"].value_counts().to_dict()

    # 1) RF Gini importance (if available)
    gini_rows: List[Dict[str, object]] = []
    est = _get_estimator(payload.pipeline)
    if hasattr(est, "feature_importances_"):
        importances = np.asarray(est.feature_importances_, dtype=float)
        for name, imp in zip(payload.feature_names, importances.tolist()):
            gini_rows.append(
                {
                    "task": "multiclass",
                    "method": "rf_gini",
                    "feature": name,
                    "importance_mean": float(imp),
                    "importance_std": 0.0,
                }
            )

    # 2) Permutation importance (multiclass)
    perm_rows: List[Dict[str, object]] = []
    perm = permutation_importance(
        payload.pipeline,
        X,
        y,
        n_repeats=int(args.permutation_repeats),
        random_state=0,
        scoring="f1_macro",
        n_jobs=-1,
    )
    for name, mean, std in zip(payload.feature_names, perm.importances_mean, perm.importances_std):
        perm_rows.append(
            {
                "task": "multiclass",
                "method": "permutation_f1_macro",
                "feature": name,
                "importance_mean": float(mean),
                "importance_std": float(std),
            }
        )

    # 3) Permutation importance for ACO vs SA only
    mask = np.isin(y, ["aco", "sa"])
    perm_rows_pair: List[Dict[str, object]] = []
    if int(mask.sum()) >= 50:
        X_pair = X[mask]
        y_pair = y[mask]
        perm_pair = permutation_importance(
            payload.pipeline,
            X_pair,
            y_pair,
            n_repeats=int(args.permutation_repeats),
            random_state=0,
            scoring="f1_macro",
            n_jobs=-1,
        )
        for name, mean, std in zip(payload.feature_names, perm_pair.importances_mean, perm_pair.importances_std):
            perm_rows_pair.append(
                {
                    "task": "aco_vs_sa",
                    "method": "permutation_f1_macro",
                    "feature": name,
                    "importance_mean": float(mean),
                    "importance_std": float(std),
                }
            )

    results = gini_rows + perm_rows + perm_rows_pair
    out_csv = out_dir / "feature_importance.csv"
    out_md = out_dir / "feature_importance.md"
    figs_dir = out_dir / "figs"

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out_csv, index=False)

    def top_table(task: str, method: str) -> str:
        sub = [r for r in results if r["task"] == task and r["method"] == method]
        if not sub:
            return "(not available)"
        sub = sorted(sub, key=lambda r: float(r["importance_mean"]), reverse=True)[: int(args.topk)]
        lines = ["| Feature | Importance (mean) | Std |", "|---|---:|---:|"]
        for r in sub:
            lines.append(
                f"| {r['feature']} | {float(r['importance_mean']):.6f} | {float(r['importance_std']):.6f} |"
            )
        return "\n".join(lines)

    md = []
    md.append("# Router Feature Importance Analysis\n")
    md.append(f"Model: `{model_path}`\n")
    md.append(f"Dataset: `{data_path}`\n")
    md.append(f"Rows: {len(df)}\n")
    md.append(f"Label counts (dataset): {label_counts}\n")
    md.append(f"SA threshold stored in model: {payload.sa_threshold}\n")

    md.append("## What this measures\n")
    md.append(
        "We report (1) tree-based Gini importance when the classifier exposes it, and (2) permutation importance "
        "using macro-F1. Higher values mean the router’s performance drops more when that feature is shuffled.\n"
    )

    md.append("## Global (multiclass) importance\n")
    md.append("### Permutation importance (macro-F1)\n")
    md.append(top_table("multiclass", "permutation_f1_macro"))
    md.append("\n")

    md.append("### RF Gini importance (if available)\n")
    md.append(top_table("multiclass", "rf_gini"))
    md.append("\n")

    md.append("## ACO vs SA (pairwise) importance\n")
    md.append(
        "This section restricts evaluation to samples whose ground-truth label is either `aco` or `sa`, then computes "
        "permutation importance on that subset. This helps interpret which features drive the ACO-vs-SA boundary.\n"
    )
    md.append(top_table("aco_vs_sa", "permutation_f1_macro"))
    md.append("\n")

    _write_markdown(out_md, "\n".join(md))

    if bool(args.save_pdfs):
        figs_dir.mkdir(parents=True, exist_ok=True)

        def save_bar(task: str, method: str, title: str, filename: str) -> None:
            sub = [r for r in results if r["task"] == task and r["method"] == method]
            if not sub:
                return
            sub = sorted(sub, key=lambda r: float(r["importance_mean"]), reverse=True)[: int(args.topk)]
            feats = [str(r["feature"]) for r in sub][::-1]
            means = [float(r["importance_mean"]) for r in sub][::-1]
            stds = [float(r["importance_std"]) for r in sub][::-1]

            plt.figure(figsize=(7.5, max(3.5, 0.28 * len(feats) + 1.2)))
            y = np.arange(len(feats))
            plt.barh(y, means, xerr=stds, capsize=2)
            plt.yticks(y, feats)
            plt.xlabel("Importance (drop in macro-F1 when permuted)")
            plt.title(title)
            plt.tight_layout()
            plt.savefig(figs_dir / filename, format="pdf")
            plt.close()

        save_bar(
            "multiclass",
            "permutation_f1_macro",
            f"Router Feature Importance (multiclass, top {int(args.topk)})",
            "feature_importance_multiclass_permutation_topk.pdf",
        )
        save_bar(
            "aco_vs_sa",
            "permutation_f1_macro",
            f"Router Feature Importance (ACO vs SA, top {int(args.topk)})",
            "feature_importance_aco_vs_sa_permutation_topk.pdf",
        )
        save_bar(
            "multiclass",
            "rf_gini",
            f"Router Feature Importance (RF Gini, top {int(args.topk)})",
            "feature_importance_rf_gini_topk.pdf",
        )

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_md}")
    if bool(args.save_pdfs):
        print(f"Wrote PDFs to: {figs_dir}")


if __name__ == "__main__":
    main()
