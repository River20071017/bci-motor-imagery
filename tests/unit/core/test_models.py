"""LDA 数组、权重读写与概率。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from eeg_core.contracts import ContractError
from eeg_core.decoding.models import (
    CLASSES,
    LdaArrays,
    fit_lda,
    load_weights,
    predict_proba,
    save_weights,
    sha256_file,
)


def _dataset() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20261103)
    features = rng.normal(size=(90, 18))
    labels = np.asarray([CLASSES[index % 3] for index in range(90)])
    return features, labels


def test_fit_predict_proba_is_normalized() -> None:
    features, labels = _dataset()
    arrays, info = fit_lda(features, labels)
    assert info["n_samples"] == 90
    probabilities = predict_proba(arrays, features[:5])
    assert probabilities.shape == (5, 3)
    assert np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_save_load_roundtrip() -> None:
    features, labels = _dataset()
    arrays, _ = fit_lda(features, labels)
    path = Path("training/models/.unit-test-model.npz")
    try:
        digest = save_weights(path, arrays)
        assert digest == sha256_file(path)

        loaded = load_weights(path)
        assert loaded.n_features == 18
        assert np.allclose(loaded.coef, arrays.coef)
        assert np.allclose(loaded.intercept, arrays.intercept)
        assert np.allclose(loaded.scaler_scale, arrays.scaler_scale)
        assert np.allclose(
            predict_proba(loaded, features[:3]), predict_proba(arrays, features[:3])
        )
    finally:
        path.unlink(missing_ok=True)


def test_fit_rejects_missing_class() -> None:
    features, _ = _dataset()
    labels = np.asarray(["idle"] * 45 + ["left_hand"] * 45)
    with pytest.raises(ContractError):
        fit_lda(features, labels)


def test_fit_rejects_too_few_per_class() -> None:
    features, _ = _dataset()
    labels = np.asarray(["idle"] * 88 + ["left_hand", "right_hand"])
    with pytest.raises(ContractError):
        fit_lda(features, labels)


def test_predict_rejects_wrong_feature_count() -> None:
    features, labels = _dataset()
    arrays, _ = fit_lda(features, labels)
    with pytest.raises(ContractError):
        predict_proba(arrays, np.zeros((1, 17)))


def test_load_rejects_missing_file() -> None:
    with pytest.raises(ContractError):
        load_weights(Path.cwd() / ".pytest-missing-model.npz")


def test_lda_arrays_reject_bad_shapes() -> None:
    with pytest.raises(ContractError):
        LdaArrays(
            coef=np.zeros((2, 18)),
            intercept=np.zeros(3),
            priors=np.ones(3) / 3,
            scaler_mean=np.zeros(18),
            scaler_scale=np.ones(18),
        )
