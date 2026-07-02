from __future__ import annotations

import argparse
import sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg
from utils.audio_utils import ensure_dir, load_audio


def pre_emphasis(y: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    if coeff <= 0:
        return y.astype(np.float32)
    if len(y) < 2:
        return y.astype(np.float32)
    return np.append(y[0], y[1:] - coeff * y[:-1]).astype(np.float32)


def cmvn(feat: np.ndarray) -> np.ndarray:
    mean = np.mean(feat, axis=0, keepdims=True)
    std = np.std(feat, axis=0, keepdims=True) + 1e-8
    return ((feat - mean) / std).astype(np.float32)


def extract_mfcc_features(
    audio_path: Path,
    sr: int = cfg.SAMPLE_RATE,
    n_mfcc: int = cfg.N_MFCC,
    n_mels: int = cfg.N_MELS,
    frame_length_ms: int = cfg.FRAME_LENGTH_MS,
    frame_shift_ms: int = cfg.FRAME_SHIFT_MS,
    include_delta: bool = cfg.INCLUDE_DELTA,
    include_delta_delta: bool = cfg.INCLUDE_DELTA_DELTA,
    use_cmvn: bool = cfg.USE_CMVN,
) -> np.ndarray:
    """Return feature matrix with shape [num_frames, num_features]."""
    y, sr = load_audio(audio_path, sr=sr)
    y = pre_emphasis(y, cfg.PREEMPHASIS)

    win_length = int(round(sr * frame_length_ms / 1000.0))
    hop_length = int(round(sr * frame_shift_ms / 1000.0))
    n_fft = 1
    while n_fft < win_length:
        n_fft *= 2

    if len(y) < win_length:
        y = np.pad(y, (0, win_length - len(y)), mode="constant")

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        n_mels=n_mels,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        window="hamming",
        center=True,
    ).T

    feats = [mfcc]
    if include_delta:
        feats.append(librosa.feature.delta(mfcc.T, order=1).T)
    if include_delta_delta:
        feats.append(librosa.feature.delta(mfcc.T, order=2).T)
    out = np.concatenate(feats, axis=1).astype(np.float32)
    if use_cmvn:
        out = cmvn(out)
    return out


def build_feature_manifest(
    segment_manifest: Path = cfg.SEGMENT_MANIFEST,
    output_dir: Path = cfg.FEATURE_NPY_DIR,
    output_manifest: Path = cfg.FEATURE_MANIFEST,
) -> pd.DataFrame:
    if not segment_manifest.exists():
        raise FileNotFoundError(f"Segment manifest not found: {segment_manifest}")
    ensure_dir(output_dir)
    ensure_dir(output_manifest.parent)

    df = pd.read_csv(segment_manifest)
    rows = []
    for idx, row in df.iterrows():
        audio_path = Path(row["path"])
        if not audio_path.is_absolute():
            audio_path = ROOT / audio_path
        feat = extract_mfcc_features(audio_path)
        feature_name = f"{idx:05d}_{Path(row['segment_filename']).stem}.npy"
        feature_path = output_dir / feature_name
        np.save(feature_path, feat)
        rows.append({
            **row.to_dict(),
            "feature_path": str(feature_path.relative_to(ROOT)),
            "num_frames": int(feat.shape[0]),
            "num_features": int(feat.shape[1]),
        })

    manifest = pd.DataFrame(rows)
    manifest.to_csv(output_manifest, index=False)
    print(f"Saved feature manifest: {output_manifest}")
    print(f"Total feature files: {len(manifest)}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract MFCC features from segmented audio.")
    parser.add_argument("--segments", type=Path, default=cfg.SEGMENT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=cfg.FEATURE_NPY_DIR)
    parser.add_argument("--manifest", type=Path, default=cfg.FEATURE_MANIFEST)
    args = parser.parse_args()
    build_feature_manifest(args.segments, args.output_dir, args.manifest)


if __name__ == "__main__":
    main()
