from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg
from utils.audio_utils import (
    ensure_dir,
    load_audio,
    overlaps_interval,
    pad_to_min_duration,
    parse_time,
    peak_normalize,
    read_annotations,
    rms_normalize,
    sanitize_filename,
    save_wav,
    slice_audio,
    trim_silence_vad,
)


def clean_segment(y: np.ndarray, sr: int, apply_vad: bool = True) -> np.ndarray:
    if cfg.PEAK_NORMALIZE:
        y = peak_normalize(y)
    else:
        y = rms_normalize(y, cfg.TARGET_DBFS)
    if apply_vad:
        y = trim_silence_vad(y, top_db=cfg.VAD_TOP_DB)
    y = pad_to_min_duration(y, sr, cfg.MIN_SEGMENT_SECONDS)
    y = peak_normalize(y)
    return y.astype(np.float32)


def create_wake_segments(
    wake_audio_path: Path,
    annotations_path: Path,
    output_dir: Path,
) -> tuple[list[dict], list[tuple[float, float]]]:
    df = read_annotations(annotations_path)
    wake_df = df[df["label"].astype(str).str.strip() == cfg.WAKE_LABEL].copy()
    if len(wake_df) == 0:
        raise ValueError(f"No rows with label={cfg.WAKE_LABEL} found in {annotations_path}")

    y, sr = load_audio(wake_audio_path, sr=cfg.SAMPLE_RATE)
    duration = len(y) / sr
    ensure_dir(output_dir)

    rows = []
    wake_intervals = []
    for idx, row in wake_df.reset_index(drop=True).iterrows():
        start = parse_time(row["start_time"])
        end = parse_time(row["end_time"])
        if start is None or end is None or end <= start:
            print(f"Skipping invalid wake annotation row {idx}: start={row['start_time']} end={row['end_time']}")
            continue

        padded_start = max(0.0, start - cfg.WAKE_PADDING_SECONDS)
        padded_end = min(duration, end + cfg.WAKE_PADDING_SECONDS)
        seg = slice_audio(y, sr, padded_start, padded_end)
        seg = clean_segment(seg, sr, apply_vad=True)

        original_name = sanitize_filename(row["filename"])
        out_name = f"wake_{idx+1:04d}_{Path(original_name).stem}.wav"
        out_path = output_dir / out_name
        save_wav(out_path, seg, sr)
        wake_intervals.append((padded_start, padded_end))
        rows.append({
            "segment_filename": out_name,
            "source_filename": str(wake_audio_path.name),
            "annotation_filename": row["filename"],
            "path": str(out_path.relative_to(ROOT)),
            "start_time": float(start),
            "end_time": float(end),
            "duration": len(seg) / sr,
            "label": cfg.WAKE_LABEL,
            "speaker": row.get("speaker", "speaker_1"),
            "environment": row.get("environment", "indoor"),
        })
    return rows, wake_intervals


def create_nonwake_segments_from_annotation(
    nonwake_audio_path: Path,
    annotations_path: Path,
    output_dir: Path,
) -> list[dict]:
    df = read_annotations(annotations_path)
    non_df = df[df["label"].astype(str).str.strip() == cfg.NON_WAKE_LABEL].copy()
    if len(non_df) == 0:
        return []

    y, sr = load_audio(nonwake_audio_path, sr=cfg.SAMPLE_RATE)
    duration = len(y) / sr
    ensure_dir(output_dir)
    rows = []
    for idx, row in non_df.reset_index(drop=True).iterrows():
        start = parse_time(row["start_time"])
        end = parse_time(row["end_time"])
        if start is None or end is None:
            continue
        if end <= start:
            continue
        seg = slice_audio(y, sr, max(0.0, start), min(duration, end))
        seg = clean_segment(seg, sr, apply_vad=False)
        out_name = f"nonwake_annot_{idx+1:04d}.wav"
        out_path = output_dir / out_name
        save_wav(out_path, seg, sr)
        rows.append({
            "segment_filename": out_name,
            "source_filename": str(nonwake_audio_path.name),
            "annotation_filename": row["filename"],
            "path": str(out_path.relative_to(ROOT)),
            "start_time": float(start),
            "end_time": float(end),
            "duration": len(seg) / sr,
            "label": cfg.NON_WAKE_LABEL,
            "speaker": row.get("speaker", "speaker_1"),
            "environment": row.get("environment", "indoor"),
        })
    return rows


def create_nonwake_segments_sliding(
    nonwake_audio_path: Path,
    output_dir: Path,
    target_count: int,
    exclude_intervals: list[tuple[float, float]] | None = None,
) -> list[dict]:
    y, sr = load_audio(nonwake_audio_path, sr=cfg.SAMPLE_RATE)
    duration = len(y) / sr
    segment_duration = cfg.NON_WAKE_SEGMENT_DURATION
    hop = cfg.NON_WAKE_SEGMENT_HOP
    if duration < segment_duration:
        raise ValueError(f"Non-wake audio is too short: {duration:.2f}s")

    exclude_intervals = exclude_intervals or []
    candidates = []
    t = 0.0
    while t + segment_duration <= duration:
        start = t
        end = t + segment_duration
        if not overlaps_interval(start, end, exclude_intervals):
            candidates.append((start, end))
        t += hop

    if not candidates:
        raise ValueError("No valid non-wake segment candidates were found.")

    rng = random.Random(cfg.RANDOM_STATE)
    rng.shuffle(candidates)
    selected = candidates[: min(target_count, len(candidates))]
    selected.sort(key=lambda x: x[0])

    ensure_dir(output_dir)
    rows = []
    for idx, (start, end) in enumerate(selected, start=1):
        seg = slice_audio(y, sr, start, end)
        # Keep non-wake context; do not trim all silence too aggressively.
        seg = clean_segment(seg, sr, apply_vad=False)
        out_name = f"nonwake_{idx:04d}_{start:.2f}s_{end:.2f}s".replace(".", "p") + ".wav"
        out_path = output_dir / out_name
        save_wav(out_path, seg, sr)
        rows.append({
            "segment_filename": out_name,
            "source_filename": str(nonwake_audio_path.name),
            "annotation_filename": "generated_from_nonwake_audio",
            "path": str(out_path.relative_to(ROOT)),
            "start_time": float(start),
            "end_time": float(end),
            "duration": len(seg) / sr,
            "label": cfg.NON_WAKE_LABEL,
            "speaker": "speaker_1",
            "environment": "indoor",
        })
    return rows


def preprocess_dataset(
    wake_audio_path: Path = cfg.WAKE_LONG_AUDIO,
    nonwake_audio_path: Path = cfg.NON_WAKE_LONG_AUDIO,
    annotations_path: Path = cfg.ANNOTATIONS_CSV,
    output_manifest: Path = cfg.SEGMENT_MANIFEST,
) -> pd.DataFrame:
    wake_audio_path = Path(wake_audio_path)
    nonwake_audio_path = Path(nonwake_audio_path)
    annotations_path = Path(annotations_path)

    if not wake_audio_path.exists():
        raise FileNotFoundError(
            f"Wake audio not found: {wake_audio_path}\n"
            "Place your 1-hour wake-word file at dataset/raw/wake_word_1hour.wav "
            "or pass --wake-audio."
        )
    if not nonwake_audio_path.exists():
        raise FileNotFoundError(
            f"Non-wake audio not found: {nonwake_audio_path}\n"
            "Place your 1-hour non-wake file at dataset/raw/non_wake_word_1hour.wav "
            "or pass --nonwake-audio."
        )

    wake_out = cfg.PROCESSED_DIR / cfg.WAKE_LABEL
    nonwake_out = cfg.PROCESSED_DIR / cfg.NON_WAKE_LABEL
    ensure_dir(wake_out)
    ensure_dir(nonwake_out)
    ensure_dir(output_manifest.parent)

    wake_rows, wake_intervals = create_wake_segments(wake_audio_path, annotations_path, wake_out)
    nonwake_rows = create_nonwake_segments_from_annotation(nonwake_audio_path, annotations_path, nonwake_out)
    if len(nonwake_rows) == 0:
        target_count = max(len(wake_rows), cfg.NON_WAKE_MAX_SEGMENTS)
        nonwake_rows = create_nonwake_segments_sliding(
            nonwake_audio_path,
            nonwake_out,
            target_count=target_count,
            exclude_intervals=[],
        )

    manifest = pd.DataFrame(wake_rows + nonwake_rows)
    manifest.to_csv(output_manifest, index=False)

    print(f"Saved segment manifest: {output_manifest}")
    print(manifest["label"].value_counts())
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess and segment 1-hour wake/non-wake audio.")
    parser.add_argument("--wake-audio", type=Path, default=cfg.WAKE_LONG_AUDIO)
    parser.add_argument("--nonwake-audio", type=Path, default=cfg.NON_WAKE_LONG_AUDIO)
    parser.add_argument("--annotations", type=Path, default=cfg.ANNOTATIONS_CSV)
    parser.add_argument("--output", type=Path, default=cfg.SEGMENT_MANIFEST)
    args = parser.parse_args()
    preprocess_dataset(args.wake_audio, args.nonwake_audio, args.annotations, args.output)


if __name__ == "__main__":
    main()
