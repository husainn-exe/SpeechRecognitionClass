"""
Central configuration for the Wake Word Detection final project.

Wake word used in this project: "Hey, Jarvis".
Modeling approach: MFCC feature extraction + HMM-GMM classification.
Deep learning models are intentionally not used.
"""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATASET_DIR = ROOT_DIR / "dataset"
RAW_DIR = DATASET_DIR / "raw"
PROCESSED_DIR = DATASET_DIR / "processed"
FEATURE_DIR = ROOT_DIR / "features"
FEATURE_NPY_DIR = FEATURE_DIR / "npy"
MODEL_DIR = ROOT_DIR / "models"
RESULT_DIR = ROOT_DIR / "results"

# Put your two 1-hour WAV files here.
WAKE_LONG_AUDIO = RAW_DIR / "wake_word_1hour.wav"
NON_WAKE_LONG_AUDIO = RAW_DIR / "non_wake_word_1hour.wav"
ANNOTATIONS_CSV = DATASET_DIR / "annotations.csv"

SEGMENT_MANIFEST = PROCESSED_DIR / "metadata_segments.csv"
FEATURE_MANIFEST = FEATURE_DIR / "features_manifest.csv"
SPLIT_MANIFEST = RESULT_DIR / "split_manifest.csv"
MODEL_PATH = MODEL_DIR / "hmm_gmm_model.pkl"

WAKE_WORD_TEXT = "Hey, Jarvis"
WAKE_LABEL = "wake_word"
NON_WAKE_LABEL = "non_wake_word"
LABELS = [WAKE_LABEL, NON_WAKE_LABEL]

# Audio preprocessing
SAMPLE_RATE = 16000
TARGET_DBFS = -20.0
PEAK_NORMALIZE = True
WAKE_PADDING_SECONDS = 0.20
VAD_TOP_DB = 30
MIN_SEGMENT_SECONDS = 0.50

# Non-wake segmentation from the 1-hour non-wake recording
NON_WAKE_SEGMENT_DURATION = 3.0
NON_WAKE_SEGMENT_HOP = 2.0
NON_WAKE_MAX_SEGMENTS = 140
RANDOM_STATE = 42

# MFCC parameters
FRAME_LENGTH_MS = 25
FRAME_SHIFT_MS = 10
N_MFCC = 13
N_MELS = 26
PREEMPHASIS = 0.97
INCLUDE_DELTA = True
INCLUDE_DELTA_DELTA = True
USE_CMVN = True

# HMM-GMM parameters
HMM_N_STATES = 5
HMM_N_MIXTURES = 4
HMM_COVARIANCE_TYPE = "diag"
HMM_N_ITER = 100
HMM_TOL = 1e-3
HMM_LEFT_RIGHT = True

# Split and threshold tuning
TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15
THRESHOLD_SELECTION = "best_f1_then_low_far"

# Long-audio scanning demo
SCAN_WINDOW_SECONDS = 3.0
SCAN_HOP_SECONDS = 1.0
DETECTION_MERGE_GAP_SECONDS = 1.0
