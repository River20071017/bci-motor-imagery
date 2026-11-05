"""窗口网格与「整窗可用」判断。"""

from __future__ import annotations

import numpy as np
import pytest

from eeg_core.contracts import DEFAULT_CONFIG, ContractError, WindowRange
from eeg_core.processing.windows import (
    hop_samples,
    is_window_usable,
    iter_window_ranges,
    required_warmup_samples,
    window_samples,
)


def test_grid_counts_match_160hz_2s_250ms() -> None:
    assert window_samples(DEFAULT_CONFIG) == 320
    assert hop_samples(DEFAULT_CONFIG) == 40
    assert required_warmup_samples(DEFAULT_CONFIG) == 480


def test_iter_window_ranges_is_anchored_and_complete() -> None:
    windows = list(iter_window_ranges(20_000, DEFAULT_CONFIG))
    assert len(windows) == 493
    assert windows[0].start_sample == 0
    assert windows[0].end_sample == 320
    assert windows[-1].start_sample == 19_680
    assert windows[-1].end_sample == 20_000
    assert all(window.is_aligned(40) for window in windows)
    assert all(window.n_samples == 320 for window in windows)


def test_short_recording_has_no_windows() -> None:
    assert list(iter_window_ranges(319, DEFAULT_CONFIG)) == []


def test_iter_window_ranges_rejects_negative_length() -> None:
    with pytest.raises(ContractError):
        list(iter_window_ranges(-1, DEFAULT_CONFIG))


def test_window_usable_requires_warmup_and_contiguous_segment() -> None:
    n = 2000
    valid = np.ones(n, dtype=bool)
    segment_start = np.zeros(n, dtype=np.int64)
    early = WindowRange(160.0, 400, 720)
    after_warmup = WindowRange(160.0, 480, 800)
    assert not is_window_usable(early, valid, segment_start, warmup=480)
    assert is_window_usable(after_warmup, valid, segment_start, warmup=480)


def test_window_usable_rejects_invalid_sample_and_segment_change() -> None:
    n = 2000
    valid = np.ones(n, dtype=bool)
    segment_start = np.zeros(n, dtype=np.int64)
    window = WindowRange(160.0, 480, 800)

    valid[600] = False
    assert not is_window_usable(window, valid, segment_start, warmup=480)

    valid[600] = True
    segment_start[700:] = 700
    assert not is_window_usable(window, valid, segment_start, warmup=480)


def test_window_usable_rejects_out_of_range() -> None:
    valid = np.ones(1000, dtype=bool)
    segment_start = np.zeros(1000, dtype=np.int64)
    window = WindowRange(160.0, 900, 1220)
    assert not is_window_usable(window, valid, segment_start, warmup=0)
