from __future__ import annotations

import argparse
import csv
import pickle
import sys
import time
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg
from features.extract_mfcc import extract_mfcc_features
from utils.audio_utils import (
    ensure_dir,
    load_audio,
    merge_intervals,
    pad_to_min_duration,
    peak_normalize,
    save_wav,
    slice_audio,
    trim_silence_vad,
)


def load_package(model_path: Path = cfg.MODEL_PATH):
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    with open(model_path, "rb") as f:
        return pickle.load(f)


def preprocess_temp_audio(input_path: Path, output_path: Path) -> Path:
    y, sr = load_audio(input_path, sr=cfg.SAMPLE_RATE)
    y = peak_normalize(y)
    y = trim_silence_vad(y, top_db=cfg.VAD_TOP_DB)
    y = pad_to_min_duration(y, sr, cfg.MIN_SEGMENT_SECONDS)
    save_wav(output_path, y, sr)
    return output_path


def plot_spectrogram(audio_path: Path, output_path: Path) -> None:
    y, sr = load_audio(audio_path, sr=cfg.SAMPLE_RATE)
    spec = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=512, hop_length=160)), ref=np.max)
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(spec, sr=sr, hop_length=160, x_axis="time", y_axis="hz", ax=ax)
    ax.set_title("Spectrogram")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def predict_audio(audio_path: Path, model_path: Path = cfg.MODEL_PATH, make_plot: bool = False) -> dict:
    ensure_dir(cfg.RESULT_DIR)
    package = load_package(model_path)
    classifier = package["classifier"]

    temp_path = cfg.RESULT_DIR / "demo_preprocessed.wav"
    preprocess_temp_audio(audio_path, temp_path)
    features = extract_mfcc_features(temp_path)

    start_time = time.perf_counter()
    pred, scores = classifier.predict(features)
    latency_ms = (time.perf_counter() - start_time) * 1000.0

    result = {
        "audio": str(audio_path),
        "wake_word": package.get("wake_word_text", cfg.WAKE_WORD_TEXT),
        "prediction": pred,
        "score": scores["score"],
        "threshold": classifier.threshold,
        "wake_norm_loglik": scores["wake_norm_loglik"],
        "nonwake_norm_loglik": scores["nonwake_norm_loglik"],
        "latency_ms": latency_ms,
    }

    print("===== Wake Word Detection Demo =====")
    print(f"Audio          : {audio_path}")
    print(f"Wake word      : {result['wake_word']}")
    print(f"Prediction     : {pred}")
    print(f"Score          : {result['score']:.6f}")
    print(f"Threshold      : {result['threshold']:.6f}")
    print(f"Latency        : {latency_ms:.2f} ms")

    if make_plot:
        out_plot = cfg.RESULT_DIR / "demo_spectrogram.png"
        plot_spectrogram(temp_path, out_plot)
        print(f"Spectrogram saved: {out_plot}")
    return result


def record_microphone(seconds: float, output_path: Path) -> Path:
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise ImportError("Microphone recording needs sounddevice. Install: pip install sounddevice") from exc
    sr = cfg.SAMPLE_RATE
    print(f"Recording {seconds:.1f} seconds. Speak the wake word if needed...")
    audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    y = audio.reshape(-1)
    save_wav(output_path, y, sr)
    print(f"Recorded audio saved: {output_path}")
    return output_path


def scan_long_audio(
    audio_path: Path,
    model_path: Path = cfg.MODEL_PATH,
    window_seconds: float = cfg.SCAN_WINDOW_SECONDS,
    hop_seconds: float = cfg.SCAN_HOP_SECONDS,
) -> Path:
    ensure_dir(cfg.RESULT_DIR)
    package = load_package(model_path)
    classifier = package["classifier"]
    y, sr = load_audio(audio_path, sr=cfg.SAMPLE_RATE)
    duration = len(y) / sr
    temp_dir = cfg.RESULT_DIR / "scan_windows"
    ensure_dir(temp_dir)

    raw_rows = []
    t = 0.0
    idx = 0
    while t + window_seconds <= duration:
        idx += 1
        seg = slice_audio(y, sr, t, t + window_seconds)
        seg = peak_normalize(seg)
        win_path = temp_dir / f"window_{idx:05d}.wav"
        save_wav(win_path, seg, sr)
        feat = extract_mfcc_features(win_path)
        pred, scores = classifier.predict(feat)
        raw_rows.append({
            "window_id": idx,
            "start_time": t,
            "end_time": t + window_seconds,
            "score": scores["score"],
            "prediction": pred,
        })
        t += hop_seconds

    raw_path = cfg.RESULT_DIR / "scan_window_scores.csv"
    with open(raw_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["window_id", "start_time", "end_time", "score", "prediction"])
        writer.writeheader()
        writer.writerows(raw_rows)

    detected = [(r["start_time"], r["end_time"]) for r in raw_rows if r["prediction"] == cfg.WAKE_LABEL]
    merged = merge_intervals(detected, gap=cfg.DETECTION_MERGE_GAP_SECONDS)
    detect_path = cfg.RESULT_DIR / "scan_detected_wake_words.csv"
    with open(detect_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["detection_id", "start_time", "end_time", "label"])
        writer.writeheader()
        for i, (s, e) in enumerate(merged, start=1):
            writer.writerow({"detection_id": i, "start_time": s, "end_time": e, "label": cfg.WAKE_LABEL})

    print(f"Window scores saved: {raw_path}")
    print(f"Merged detections saved: {detect_path}")
    print(f"Total detections: {len(merged)}")
    return detect_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo HMM-GMM wake-word detection.")
    parser.add_argument("--audio", type=Path, help="Audio file to classify.")
    parser.add_argument("--model", type=Path, default=cfg.MODEL_PATH)
    parser.add_argument("--plot", action="store_true", help="Save spectrogram image.")
    parser.add_argument("--record", type=float, default=None, help="Record microphone for N seconds, then classify.")
    parser.add_argument("--scan-long", action="store_true", help="Scan a long audio using sliding windows.")
    parser.add_argument("--window", type=float, default=cfg.SCAN_WINDOW_SECONDS)
    parser.add_argument("--hop", type=float, default=cfg.SCAN_HOP_SECONDS)
    args = parser.parse_args()

    if args.record is not None:
        audio_path = cfg.RESULT_DIR / "recorded_demo.wav"
        ensure_dir(cfg.RESULT_DIR)
        record_microphone(args.record, audio_path)
    elif args.audio is not None:
        audio_path = args.audio
    else:
        raise SystemExit("Provide --audio path/to/file.wav or --record N")

    if args.scan_long:
        scan_long_audio(audio_path, args.model, args.window, args.hop)
    else:
        predict_audio(audio_path, args.model, args.plot)


if __name__ == "__main__":
    main()
