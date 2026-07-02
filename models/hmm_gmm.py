from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np


def make_left_right_transmat(n_states: int) -> np.ndarray:
    """Create a Bakis/left-to-right transition matrix."""
    transmat = np.zeros((n_states, n_states), dtype=np.float64)
    for i in range(n_states):
        if i == n_states - 1:
            transmat[i, i] = 1.0
        else:
            transmat[i, i] = 0.60
            transmat[i, i + 1] = 0.40
    return transmat


def make_startprob(n_states: int) -> np.ndarray:
    startprob = np.zeros(n_states, dtype=np.float64)
    startprob[0] = 1.0
    return startprob


@dataclass
class HMMGMMClassifier:
    """Two-model HMM-GMM classifier: wake-word model vs non-wake model."""

    wake_model: object
    nonwake_model: object
    scaler: object
    threshold: float
    wake_label: str = "wake_word"
    nonwake_label: str = "non_wake_word"

    def transform(self, features: np.ndarray) -> np.ndarray:
        return self.scaler.transform(features)

    def score_pair(self, features: np.ndarray) -> dict:
        x = self.transform(features)
        wake_loglik = float(self.wake_model.score(x))
        nonwake_loglik = float(self.nonwake_model.score(x))
        n = max(len(x), 1)
        wake_norm = wake_loglik / n
        nonwake_norm = nonwake_loglik / n
        score = wake_norm - nonwake_norm
        return {
            "wake_loglik": wake_loglik,
            "nonwake_loglik": nonwake_loglik,
            "wake_norm_loglik": wake_norm,
            "nonwake_norm_loglik": nonwake_norm,
            "score": score,
        }

    def predict(self, features: np.ndarray) -> tuple[str, dict]:
        scores = self.score_pair(features)
        pred = self.wake_label if scores["score"] >= self.threshold else self.nonwake_label
        scores["prediction"] = pred
        return pred, scores


def create_gmm_hmm(
    n_states: int,
    n_mixtures: int,
    covariance_type: str = "diag",
    n_iter: int = 100,
    tol: float = 1e-3,
    random_state: int = 42,
    left_right: bool = True,
):
    """Create hmmlearn GMMHMM with optional left-to-right topology."""
    try:
        from hmmlearn.hmm import GMMHMM
    except ImportError as exc:
        raise ImportError(
            "hmmlearn is required for HMM-GMM training. Install it with: pip install hmmlearn"
        ) from exc

    model = GMMHMM(
        n_components=n_states,
        n_mix=n_mixtures,
        covariance_type=covariance_type,
        n_iter=n_iter,
        tol=tol,
        random_state=random_state,
        verbose=False,
        min_covar=1e-3,
    )

    if left_right:
        model.startprob_ = make_startprob(n_states)
        model.transmat_ = make_left_right_transmat(n_states)
        # Keep start/transition initialized as Bakis topology; learn emission GMM parameters.
        model.init_params = "mcw"
        model.params = "mcw"
    return model


def fit_hmm_gmm(model, sequences: list[np.ndarray]):
    if not sequences:
        raise ValueError("No sequences provided for HMM-GMM training.")
    lengths = [len(x) for x in sequences]
    x_concat = np.vstack(sequences)
    model.fit(x_concat, lengths)
    return model
