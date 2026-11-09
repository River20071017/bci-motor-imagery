"""18 维对数方差特征的形状、顺序与下限。"""

from __future__ import annotations

import numpy as np
import pytest

from eeg_core.contracts import ContractError
from eeg_core.processing.features import FEATURE_DIM, log_variance_features


def _bands(n_channels: int = 3, n_samples: int = 320) -> tuple[np.ndarray, np.ndarray]:
    time = np.arange(n_samples) / 160.0
    first = np.zeros((n_channels, n_samples))
    second = np.zeros((n_channels, n_samples))
    for index in range(n_channels):
        first[index] = (index + 1) * 1e-6 * np.sin(2 * np.pi * 10 * time)
        second[index] = (index + 1) * 2e-6 * np.sin(2 * np.pi * 20 * time)
    return first, second


def test_feature_dim_and_order() -> None:
    first, second = _bands()
    features = log_variance_features(
        first, second, [2, 0], 0, 320, floor=1e-15
    )
    assert features.shape == (4,)
    assert FEATURE_DIM == 18

    # channel_band：先通道 2 的两带，再通道 0 的两带
    expected_ch2_first = np.log(np.var(first[2], ddof=0))
    expected_ch2_second = np.log(np.var(second[2], ddof=0))
    expected_ch0_first = np.log(np.var(first[0], ddof=0))
    expected_ch0_second = np.log(np.var(second[0], ddof=0))
    assert features[0] == pytest.approx(expected_ch2_first)
    assert features[1] == pytest.approx(expected_ch2_second)
    assert features[2] == pytest.approx(expected_ch0_first)
    assert features[3] == pytest.approx(expected_ch0_second)


def test_floor_applies_to_constant_window() -> None:
    first = np.zeros((1, 320))
    second = np.zeros((1, 320))
    features = log_variance_features(first, second, [0], 0, 320, floor=1e-12)
    assert features[0] == pytest.approx(np.log(1e-12))
    assert features[1] == pytest.approx(np.log(1e-12))


def test_ddof_changes_variance() -> None:
    first, second = _bands(1, 320)
    population = log_variance_features(first, second, [0], 0, 320, floor=1e-12)
    sample = log_variance_features(first, second, [0], 0, 320, floor=1e-12, ddof=1)
    assert not np.allclose(population, sample)


@pytest.mark.parametrize(
    ("start", "end"),
    [(-1, 320), (0, 321), (100, 100), (200, 100)],
)
def test_bad_window_rejected(start: int, end: int) -> None:
    first, second = _bands(1, 320)
    with pytest.raises(ContractError):
        log_variance_features(first, second, [0], start, end, floor=1e-12)


def test_shape_mismatch_rejected() -> None:
    first, second = _bands(2, 320)
    with pytest.raises(ContractError):
        log_variance_features(first, second[:, :100], [0], 0, 100, floor=1e-12)


def test_bad_channel_index_rejected() -> None:
    first, second = _bands(2, 320)
    with pytest.raises(ContractError):
        log_variance_features(first, second, [2], 0, 320, floor=1e-12)
