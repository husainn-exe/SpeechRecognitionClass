from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import librosa
import numpy as np
import pandas as pd
import soundfile as sf


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sanitize_filename(name: str) -> str:
    """Create a safe filename while preserving readability."""
    name = str(name).strip().replace(" ", "_")
    name = re.sub(r"[^A-Za-z0-9_.()\-]+", "_", name)
    return name.strip("_") or "audio"


def parse_time(value) -> float | None:
    """Parse annotation timestamps. Returns None for '-', blank, or invalid values."""
    if value is None:
        return None
    text = str(value).strip()
    if text in {"", "-", "nan", "None"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def load_audio(path: Path, sr: int = 16000) -> tuple[np.ndarray, int]:
    if not Path(path).exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    y, loaded_sr = librosa.load(str(path), sr=sr, mono=True)
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        raise ValueError(f"Audio file is empty: {path}")
    return y, sr


def save_wav(path: Path, y: np.ndarray, sr: int = 16000) -> None:
    ensure_dir(Path(path).parent)
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        raise ValueError(f"Cannot save empty audio: {path}")
    y = np.clip(y, -1.0, 1.0)
    sf.write(str(path), y, sr, subtype="PCM_16")


def peak_normalize(y: np.ndarray, peak: float = 0.98) -> np.ndarray:
    max_abs = float(np.max(np.abs(y))) if y.size else 0.0
    if max_abs < 1e-9:
        return y.astype(np.float32)
    return (y / max_abs * peak).astype(np.float32)


def rms_dbfs(y: np.ndarray) -> float:
    rms = np.sqrt(np.mean(np.square(y)) + 1e-12)
    return 20.0 * np.log10(rms + 1e-12)


def rms_normalize(y: np.ndarray, target_dbfs: float = -20.0) -> np.ndarray:
    current = rms_dbfs(y)
    gain = 10.0 ** ((target_dbfs - current) / 20.0)
    out = y * gain
    # Avoid clipping after RMS normalization.
    max_abs = float(np.max(np.abs(out))) if out.size else 0.0
    if max_abs > 0.99:
        out = out / max_abs * 0.99
    return out.astype(np.float32)


def trim_silence_vad(y: np.ndarray, top_db: int = 30) -> np.ndarray:
    """Simple VAD using librosa.effects.split. Keeps only non-silent intervals."""
    if y.size == 0:
        return y
    intervals = librosa.effects.split(y, top_db=top_db)
    if len(intervals) == 0:
        return y.astype(np.float32)
    chunks = [y[start:end] for start, end in intervals if end > start]
    if not chunks:
        return y.astype(np.float32)
    return np.concatenate(chunks).astype(np.float32)


def pad_to_min_duration(y: np.ndarray, sr: int, min_seconds: float) -> np.ndarray:
    min_len = int(round(min_seconds * sr))
    if len(y) >= min_len:
        return y.astype(np.float32)
    pad_left = (min_len - len(y)) // 2
    pad_right = min_len - len(y) - pad_left
    return np.pad(y, (pad_left, pad_right), mode="constant").astype(np.float32)


def slice_audio(y: np.ndarray, sr: int, start_s: float, end_s: float) -> np.ndarray:
    start = max(0, int(round(start_s * sr)))
    end = min(len(y), int(round(end_s * sr)))
    if end <= start:
        raise ValueError(f"Invalid segment start/end: {start_s} - {end_s}")
    return y[start:end].astype(np.float32)


def read_annotations(path: Path) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"Annotation file not found: {path}")
    df = pd.read_csv(path)
    required = {"filename", "start_time", "end_time", "label", "speaker", "environment"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in annotation CSV: {sorted(missing)}")
    df = df.copy()
    df["label"] = df["label"].astype(str).str.strip()
    return df


def merge_intervals(intervals: Sequence[tuple[float, float]], gap: float = 0.0) -> list[tuple[float, float]]:
    valid = sorted((float(s), float(e)) for s, e in intervals if e > s)
    if not valid:
        return []
    merged = [valid[0]]
    for s, e in valid[1:]:
        last_s, last_e = merged[-1]
        if s <= last_e + gap:
            merged[-1] = (last_s, max(last_e, e))
        else:
            merged.append((s, e))
    return merged


def overlaps_interval(start: float, end: float, intervals: Sequence[tuple[float, float]]) -> bool:
    for s, e in intervals:
        if start < e and end > s:
            return True
    return False
