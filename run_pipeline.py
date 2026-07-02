from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    py = sys.executable
    run([py, "preprocessing/preprocess_audio.py"])
    run([py, "features/extract_mfcc.py"])
    run([py, "models/train_hmm_gmm.py"])
    run([py, "evaluation/evaluate.py"])
    print("\nPipeline finished. Check the models/ and results/ folders.")


if __name__ == "__main__":
    main()
