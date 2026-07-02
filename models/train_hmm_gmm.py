from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg
from models.hmm_gmm import HMMGMMClassifier, create_gmm_hmm, fit_hmm_gmm
from utils.audio_utils import ensure_dir


def load_feature(feature_path: str | Path) -> np.ndarray:
    path = Path(feature_path)
    if not path.is_absolute():
        path = ROOT / path
    return np.load(path).astype(np.float32)


def stratified_three_way_split(df: pd.DataFrame) -> pd.DataFrame:
    """Add split column: train / validation / test."""
    if "label" not in df.columns:
        raise ValueError("Feature manifest must contain a label column.")

    train_df, temp_df = train_test_split(
        df,
        train_size=cfg.TRAIN_SIZE,
        stratify=df["label"],
        random_state=cfg.RANDOM_STATE,
        shuffle=True,
    )

    val_fraction_of_temp = cfg.VALIDATION_SIZE / (cfg.VALIDATION_SIZE + cfg.TEST_SIZE)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_fraction_of_temp,
        stratify=temp_df["label"],
        random_state=cfg.RANDOM_STATE,
        shuffle=True,
    )

    train_df = train_df.copy(); train_df["split"] = "train"
    val_df = val_df.copy(); val_df["split"] = "validation"
    test_df = test_df.copy(); test_df["split"] = "test"
    split_df = pd.concat([train_df, val_df, test_df], axis=0).sort_index().reset_index(drop=True)
    return split_df


def fit_scaler(train_df: pd.DataFrame) -> StandardScaler:
    frames = [load_feature(p) for p in train_df["feature_path"]]
    x = np.vstack(frames)
    scaler = StandardScaler()
    scaler.fit(x)
    return scaler


def load_sequences(df: pd.DataFrame, scaler: StandardScaler | None = None) -> list[np.ndarray]:
    seqs = []
    for p in df["feature_path"]:
        x = load_feature(p)
        if scaler is not None:
            x = scaler.transform(x)
        seqs.append(x.astype(np.float32))
    return seqs


def compute_scores(classifier: HMMGMMClassifier, df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        x = load_feature(row["feature_path"])
        pred, scores = classifier.predict(x)
        rows.append({**row.to_dict(), **scores})
    return pd.DataFrame(rows)


def metrics_from_predictions(y_true: list[str], y_pred: list[str]) -> dict:
    tp = sum((t == cfg.WAKE_LABEL and p == cfg.WAKE_LABEL) for t, p in zip(y_true, y_pred))
    fn = sum((t == cfg.WAKE_LABEL and p == cfg.NON_WAKE_LABEL) for t, p in zip(y_true, y_pred))
    fp = sum((t == cfg.NON_WAKE_LABEL and p == cfg.WAKE_LABEL) for t, p in zip(y_true, y_pred))
    tn = sum((t == cfg.NON_WAKE_LABEL and p == cfg.NON_WAKE_LABEL) for t, p in zip(y_true, y_pred))

    def safe_div(a, b):
        return float(a / b) if b else 0.0

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    accuracy = safe_div(tp + tn, tp + tn + fp + fn)
    far = safe_div(fp, fp + tn)
    frr = safe_div(fn, fn + tp)
    return {
        "TP": tp,
        "FN": fn,
        "FP": fp,
        "TN": tn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "FAR": far,
        "FRR": frr,
    }


def tune_threshold(classifier: HMMGMMClassifier, val_df: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    scored = compute_scores(classifier, val_df)
    scores = np.sort(scored["score"].values.astype(float))
    if len(scores) == 0:
        raise ValueError("Validation set is empty; cannot tune threshold.")

    candidates = []
    candidates.append(scores[0] - 1e-6)
    candidates.append(scores[-1] + 1e-6)
    if len(scores) > 1:
        candidates.extend(((scores[:-1] + scores[1:]) / 2.0).tolist())
    candidates = sorted(set(float(x) for x in candidates))

    rows = []
    for th in candidates:
        pred = [cfg.WAKE_LABEL if s >= th else cfg.NON_WAKE_LABEL for s in scored["score"]]
        met = metrics_from_predictions(scored["label"].tolist(), pred)
        rows.append({"threshold": th, **met})
    metrics_df = pd.DataFrame(rows)
    # Prioritize F1, then lower FAR, then lower FRR, then higher accuracy.
    best = metrics_df.sort_values(
        by=["f1_score", "FAR", "FRR", "accuracy"],
        ascending=[False, True, True, False],
    ).iloc[0]
    return float(best["threshold"]), metrics_df


def train_model(feature_manifest: Path = cfg.FEATURE_MANIFEST, model_path: Path = cfg.MODEL_PATH) -> HMMGMMClassifier:
    if not feature_manifest.exists():
        raise FileNotFoundError(f"Feature manifest not found: {feature_manifest}")
    ensure_dir(cfg.RESULT_DIR)
    ensure_dir(model_path.parent)

    df = pd.read_csv(feature_manifest)
    required_labels = {cfg.WAKE_LABEL, cfg.NON_WAKE_LABEL}
    present_labels = set(df["label"].unique())
    missing = required_labels.difference(present_labels)
    if missing:
        raise ValueError(f"Missing required labels in manifest: {missing}")

    split_df = stratified_three_way_split(df)
    split_df.to_csv(cfg.SPLIT_MANIFEST, index=False)
    print(f"Saved split manifest: {cfg.SPLIT_MANIFEST}")
    print(split_df.groupby(["split", "label"]).size())

    train_df = split_df[split_df["split"] == "train"].copy()
    val_df = split_df[split_df["split"] == "validation"].copy()

    scaler = fit_scaler(train_df)
    wake_train = train_df[train_df["label"] == cfg.WAKE_LABEL]
    nonwake_train = train_df[train_df["label"] == cfg.NON_WAKE_LABEL]

    wake_model = create_gmm_hmm(
        cfg.HMM_N_STATES,
        cfg.HMM_N_MIXTURES,
        cfg.HMM_COVARIANCE_TYPE,
        cfg.HMM_N_ITER,
        cfg.HMM_TOL,
        cfg.RANDOM_STATE,
        cfg.HMM_LEFT_RIGHT,
    )
    nonwake_model = create_gmm_hmm(
        cfg.HMM_N_STATES,
        cfg.HMM_N_MIXTURES,
        cfg.HMM_COVARIANCE_TYPE,
        cfg.HMM_N_ITER,
        cfg.HMM_TOL,
        cfg.RANDOM_STATE + 7,
        cfg.HMM_LEFT_RIGHT,
    )

    print("Training wake-word HMM-GMM...")
    fit_hmm_gmm(wake_model, load_sequences(wake_train, scaler))
    print("Training non-wake HMM-GMM...")
    fit_hmm_gmm(nonwake_model, load_sequences(nonwake_train, scaler))

    classifier = HMMGMMClassifier(
        wake_model=wake_model,
        nonwake_model=nonwake_model,
        scaler=scaler,
        threshold=0.0,
        wake_label=cfg.WAKE_LABEL,
        nonwake_label=cfg.NON_WAKE_LABEL,
    )

    threshold, threshold_df = tune_threshold(classifier, val_df)
    classifier.threshold = threshold
    threshold_path = cfg.RESULT_DIR / "validation_threshold_analysis.csv"
    threshold_df.to_csv(threshold_path, index=False)
    print(f"Selected threshold: {threshold:.6f}")
    print(f"Saved validation threshold analysis: {threshold_path}")

    package = {
        "classifier": classifier,
        "wake_word_text": cfg.WAKE_WORD_TEXT,
        "sample_rate": cfg.SAMPLE_RATE,
        "mfcc": {
            "n_mfcc": cfg.N_MFCC,
            "n_mels": cfg.N_MELS,
            "frame_length_ms": cfg.FRAME_LENGTH_MS,
            "frame_shift_ms": cfg.FRAME_SHIFT_MS,
            "include_delta": cfg.INCLUDE_DELTA,
            "include_delta_delta": cfg.INCLUDE_DELTA_DELTA,
            "use_cmvn": cfg.USE_CMVN,
        },
        "hmm_gmm": {
            "n_states": cfg.HMM_N_STATES,
            "n_mixtures": cfg.HMM_N_MIXTURES,
            "covariance_type": cfg.HMM_COVARIANCE_TYPE,
            "left_right": cfg.HMM_LEFT_RIGHT,
        },
    }
    with open(model_path, "wb") as f:
        pickle.dump(package, f)
    print(f"Saved model package: {model_path}")
    return classifier


def main() -> None:
    parser = argparse.ArgumentParser(description="Train wake-word HMM-GMM model.")
    parser.add_argument("--features", type=Path, default=cfg.FEATURE_MANIFEST)
    parser.add_argument("--model", type=Path, default=cfg.MODEL_PATH)
    args = parser.parse_args()
    train_model(args.features, args.model)


if __name__ == "__main__":
    main()
