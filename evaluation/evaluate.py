from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg
from models.train_hmm_gmm import load_feature, metrics_from_predictions
from utils.audio_utils import ensure_dir


def load_classifier(model_path: Path = cfg.MODEL_PATH):
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    with open(model_path, "rb") as f:
        package = pickle.load(f)
    return package["classifier"], package


def predict_dataframe(classifier, df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        x = load_feature(row["feature_path"])
        pred, scores = classifier.predict(x)
        rows.append({**row.to_dict(), **scores})
    return pd.DataFrame(rows)


def save_confusion_matrix(y_true, y_pred, output_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[cfg.WAKE_LABEL, cfg.NON_WAKE_LABEL])
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Wake Word", "Non-Wake Word"],
    )
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title("Confusion Matrix - Wake Word Detection")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def threshold_sweep(scored: pd.DataFrame) -> pd.DataFrame:
    scores = np.asarray(scored["score"], dtype=float)
    if len(scores) == 0:
        return pd.DataFrame()
    thresholds = np.linspace(scores.min() - 1e-6, scores.max() + 1e-6, 50)
    rows = []
    y_true = scored["label"].tolist()
    for th in thresholds:
        pred = [cfg.WAKE_LABEL if s >= th else cfg.NON_WAKE_LABEL for s in scores]
        rows.append({"threshold": float(th), **metrics_from_predictions(y_true, pred)})
    return pd.DataFrame(rows)


def save_threshold_plot(sweep: pd.DataFrame, output_path: Path) -> None:
    if sweep.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    for metric in ["accuracy", "precision", "recall", "f1_score", "FAR", "FRR"]:
        ax.plot(sweep["threshold"], sweep[metric], label=metric)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Metric value")
    ax.set_title("Threshold Analysis")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def evaluate(model_path: Path = cfg.MODEL_PATH, split_manifest: Path = cfg.SPLIT_MANIFEST) -> dict:
    ensure_dir(cfg.RESULT_DIR)
    classifier, package = load_classifier(model_path)
    if not split_manifest.exists():
        raise FileNotFoundError(f"Split manifest not found: {split_manifest}")
    df = pd.read_csv(split_manifest)
    test_df = df[df["split"] == "test"].copy()
    if test_df.empty:
        raise ValueError("No test rows found in split manifest.")

    scored = predict_dataframe(classifier, test_df)
    predictions_path = cfg.RESULT_DIR / "predictions_test.csv"
    scored.to_csv(predictions_path, index=False)

    metrics = metrics_from_predictions(scored["label"].tolist(), scored["prediction"].tolist())
    metrics["threshold"] = classifier.threshold
    metrics_df = pd.DataFrame([metrics])
    metrics_path = cfg.RESULT_DIR / "evaluation_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    cm_path = cfg.RESULT_DIR / "confusion_matrix.png"
    save_confusion_matrix(scored["label"], scored["prediction"], cm_path)

    sweep = threshold_sweep(scored)
    sweep_path = cfg.RESULT_DIR / "threshold_analysis.csv"
    sweep.to_csv(sweep_path, index=False)
    save_threshold_plot(sweep, cfg.RESULT_DIR / "threshold_analysis.png")

    print("Evaluation metrics:")
    print(metrics_df.T)
    print(f"Saved: {metrics_path}")
    print(f"Saved: {cm_path}")
    print(f"Saved: {predictions_path}")
    print(f"Saved: {sweep_path}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate HMM-GMM wake-word detection model.")
    parser.add_argument("--model", type=Path, default=cfg.MODEL_PATH)
    parser.add_argument("--split", type=Path, default=cfg.SPLIT_MANIFEST)
    args = parser.parse_args()
    evaluate(args.model, args.split)


if __name__ == "__main__":
    main()
