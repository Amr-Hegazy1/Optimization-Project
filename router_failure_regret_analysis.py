from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LABELS_DEFAULT = ["sa", "ga", "aco"]
COST_COLS = {"sa": "sa_cost", "ga": "ga_cost", "aco": "aco_cost"}


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

    return ModelPayload(
        pipeline=payload,
        feature_names=[],
        allowed_keys=LABELS_DEFAULT,
        sa_threshold=None,
    )


def _get_estimator(pipeline: Any) -> Any:
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

    for k, c in COST_COLS.items():
        if c not in df.columns:
            raise SystemExit(f"Dataset missing cost column: {c}")

    X = df[feat_cols].to_numpy(dtype=float)
    y = df["label"].astype(str).to_numpy(dtype=object)
    return X, y, df


def _write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _quantiles(x: np.ndarray) -> Dict[str, float]:
    if x.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "p90": float("nan"), "p99": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p90": float(np.quantile(x, 0.90)),
        "p99": float(np.quantile(x, 0.99)),
        "max": float(np.max(x)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Failure mode + regret analysis for the SA/GA/ACO router.")
    p.add_argument("--model", type=str, default="runs/router/model.joblib")
    p.add_argument("--data", type=str, default="runs/router/dataset_combined_min_cost.csv")
    p.add_argument("--out-dir", type=str, default="runs/router/analysis")
    p.add_argument("--labels", type=str, default=",".join(LABELS_DEFAULT), help="Label order for confusion matrix.")
    p.add_argument(
        "--save-pdfs",
        action="store_true",
        help="If set, save publication-ready PDF figures to <out-dir>/figs.",
    )
    args = p.parse_args()

    model_path = Path(args.model)
    data_path = Path(args.data)
    out_dir = Path(args.out_dir)
    labels = [s.strip() for s in str(args.labels).split(",") if s.strip()]

    payload = _load_payload(model_path)
    if not payload.feature_names:
        raise SystemExit(
            "Model does not include feature_names. Re-train using the provided router_train.py to embed them."
        )

    X, y_true, df = _load_dataset(data_path, payload.feature_names)
    y_pred = predict_with_optional_sa_threshold(payload.pipeline, X, payload.sa_threshold)

    # Confusion matrix + report
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(y_true, y_pred, digits=4, zero_division=0)

    # Safety slice: ACO ground truth predicted as SA
    y_true_s = pd.Series(y_true)
    y_pred_s = pd.Series(y_pred)

    aco_to_sa = int(((y_true_s == "aco") & (y_pred_s == "sa")).sum())
    aco_total = int((y_true_s == "aco").sum())
    aco_to_sa_rate = (aco_to_sa / aco_total) if aco_total else 0.0

    # Regret: predicted cost minus best-achievable cost among {sa,ga,aco}
    costs = df[["sa_cost", "ga_cost", "aco_cost"]].to_numpy(dtype=float)
    min_cost = np.min(costs, axis=1)

    pred_cost = np.full(shape=(len(df),), fill_value=np.nan, dtype=float)
    for k, col in COST_COLS.items():
        mask = (y_pred_s == k).to_numpy()
        pred_cost[mask] = df.loc[mask, col].to_numpy(dtype=float)

    regret = pred_cost - min_cost
    rel_regret = regret / np.maximum(1e-12, np.abs(min_cost))

    # Aggregate regret overall and by true label
    overall = _quantiles(regret)

    by_true: List[Dict[str, object]] = []
    for lab in labels:
        m = (y_true_s == lab).to_numpy()
        q = _quantiles(regret[m])
        by_true.append({"true_label": lab, **q, "count": int(np.sum(m))})

    # Specific safety regret slice (true=aco, pred=sa)
    m_safety = ((y_true_s == "aco") & (y_pred_s == "sa")).to_numpy()
    safety = _quantiles(regret[m_safety])

    out_dir.mkdir(parents=True, exist_ok=True)

    # Write per-sample regret CSV
    out_regret_csv = out_dir / "regret_per_sample.csv"
    out_agg_csv = out_dir / "regret_by_true_label.csv"
    out_md = out_dir / "failure_and_regret.md"
    figs_dir = out_dir / "figs"

    per_sample = pd.DataFrame(
        {
            "seed": df.get("seed", pd.Series([None] * len(df))),
            "budget": df.get("budget", pd.Series([None] * len(df))),
            "y_true": y_true_s,
            "y_pred": y_pred_s,
            "min_cost": min_cost,
            "pred_cost": pred_cost,
            "regret": regret,
            "rel_regret": rel_regret,
        }
    )
    per_sample.to_csv(out_regret_csv, index=False)
    pd.DataFrame(by_true).to_csv(out_agg_csv, index=False)

    # Render confusion matrix as markdown table
    cm_lines = ["| True\\Pred | " + " | ".join(labels) + " |", "|---|" + "---:|" * len(labels)]
    for i, lab in enumerate(labels):
        cm_lines.append("| " + lab + " | " + " | ".join(str(int(x)) for x in cm[i, :]) + " |")

    md: List[str] = []
    md.append("# Router Failure Mode & Regret Analysis\n")
    md.append(f"Model: `{model_path}`\n")
    md.append(f"Dataset: `{data_path}`\n")
    md.append(f"Rows: {len(df)}\n")
    md.append(f"SA threshold stored in model: {payload.sa_threshold}\n")

    md.append("## Confusion matrix\n")
    md.extend(cm_lines)
    md.append("\n")

    md.append("## Classification report\n")
    md.append("```\n" + report + "\n```\n")

    md.append("## Safety subsection: ACO ground truth misrouted to SA\n")
    md.append(
        f"Count(true=aco, pred=sa): **{aco_to_sa} / {aco_total}** (rate={aco_to_sa_rate:.6f}).\n"
    )
    md.append(
        "Interpretation: if this count is 0 (or extremely low), the router is conservative about selecting SA when ACO is truly optimal by cost.\n"
    )

    md.append("## Regret (cost overhead vs best-cost oracle)\n")
    md.append(
        f"Overall regret summary: mean={overall['mean']:.6f}, median={overall['median']:.6f}, p90={overall['p90']:.6f}, p99={overall['p99']:.6f}, max={overall['max']:.6f}.\n"
    )

    md.append("### Regret by true label\n")
    md.append("| True label | Count | Mean | Median | P90 | P99 | Max |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in by_true:
        md.append(
            "| {true_label} | {count} | {mean:.6f} | {median:.6f} | {p90:.6f} | {p99:.6f} | {max:.6f} |".format(
                **r
            )
        )
    md.append("\n")

    md.append("### Safety regret slice (true=aco, pred=sa)\n")
    md.append(
        f"If any cases exist, regret summary: mean={safety['mean']:.6f}, median={safety['median']:.6f}, p90={safety['p90']:.6f}, p99={safety['p99']:.6f}, max={safety['max']:.6f}.\n"
    )

    _write_markdown(out_md, "\n".join(md))

    if bool(args.save_pdfs):
        figs_dir.mkdir(parents=True, exist_ok=True)

        # Confusion matrix heatmap
        plt.figure(figsize=(5.2, 4.6))
        plt.imshow(cm, interpolation="nearest")
        plt.title("Router Confusion Matrix")
        plt.colorbar(fraction=0.046, pad=0.04)
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, rotation=45, ha="right")
        plt.yticks(tick_marks, labels)
        plt.ylabel("True")
        plt.xlabel("Predicted")
        # annotate counts
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(int(cm[i, j])), ha="center", va="center")
        plt.tight_layout()
        plt.savefig(figs_dir / "confusion_matrix.pdf", format="pdf")
        plt.close()

        # Regret histogram (overall)
        r = regret[np.isfinite(regret)]
        plt.figure(figsize=(6.5, 4.2))
        # show heavy tail safely by clipping for the histogram view
        clip = float(np.quantile(r, 0.99)) if r.size else 0.0
        r_clip = np.clip(r, a_min=None, a_max=clip)
        plt.hist(r_clip, bins=50)
        plt.title("Router Regret Histogram (clipped at p99)")
        plt.xlabel("Regret = cost(pred) - min(cost)")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(figs_dir / "regret_hist_p99_clipped.pdf", format="pdf")
        plt.close()

        # Regret by true label (boxplot)
        plt.figure(figsize=(6.5, 4.2))
        data = []
        for lab in labels:
            m = (y_true_s == lab).to_numpy()
            data.append(regret[m])
        plt.boxplot(data, labels=labels, showfliers=False)
        plt.title("Regret by True Label (no fliers)")
        plt.ylabel("Regret")
        plt.tight_layout()
        plt.savefig(figs_dir / "regret_by_true_label_boxplot.pdf", format="pdf")
        plt.close()

        # Safety figure: ACO->SA rate
        plt.figure(figsize=(5.8, 3.6))
        plt.bar(["true=ACO→pred=SA"], [aco_to_sa_rate])
        plt.ylim(0.0, max(aco_to_sa_rate * 1.4, 0.001))
        plt.title("Safety: ACO misrouted to SA")
        plt.ylabel("Rate")
        plt.tight_layout()
        plt.savefig(figs_dir / "safety_aco_to_sa_rate.pdf", format="pdf")
        plt.close()

    print(f"Wrote: {out_md}")
    print(f"Wrote: {out_regret_csv}")
    print(f"Wrote: {out_agg_csv}")
    if bool(args.save_pdfs):
        print(f"Wrote PDFs to: {figs_dir}")


if __name__ == "__main__":
    main()
