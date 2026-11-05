"""2a 数据源与「T 内 run CV / E 封存」协议的可验证部分。

用内存里合成的 2a 结构（6 run × 48 trial 的缩水版）验证：类别与 idle 标记、单位换算、
E session 封存、按 run 分组。真实下载脚本见 ``scripts/download_bnci2014.py``。
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from eeg_app.sources.bnci2014 import (
    EEG_CHANNELS_2A,
    EOG_CHANNELS_2A,
    load_bnci2014_session,
    parse_bnci2014_mat,
)
from eeg_core.contracts import DEFAULT_CONFIG, AppConfig, ContractError, WindowRange
from training.cv import run_grouped_cv, window_trial_and_run
from training.pipeline import label_of_window

RATE = 250.0
TRIAL_SEC = 7.5
CUE_SEC = 2.0


def _run(classes: list[int], *, seconds: float | None = None) -> dict[str, object]:
    if seconds is None:
        seconds = CUE_SEC + TRIAL_SEC * (len(classes) - 1) + 6.0
    n_samples = int(round(seconds * RATE))
    # 用噪声而不是常量：常量会被质量门判 flatline，CV 测试就没有窗了。
    rng = np.random.default_rng(20261105)
    x_uv = rng.normal(0.0, 5.0, size=(n_samples, 25))
    trial = np.asarray(
        [int(round((CUE_SEC + TRIAL_SEC * index) * RATE)) + 1 for index in range(len(classes))],
        dtype=np.int64,
    )
    return {"X": x_uv, "fs": RATE, "trial": trial, "y": np.asarray(classes, dtype=np.int64)}


def _recording(classes: list[int], *, subject: int = 1, session: str = "T"):
    return parse_bnci2014_mat([_run(classes)], subject=subject, session=session)


def test_parse_maps_classes_and_idle() -> None:
    recording = _recording([1, 2])
    assert recording.info.sample_rate_hz == RATE
    assert len(recording.info.channel_names) == 25
    assert set(EEG_CHANNELS_2A).issubset(recording.info.channel_names)
    assert set(EOG_CHANNELS_2A).issubset(recording.info.channel_names)
    assert recording.info.subject_id == "A01"
    assert recording.info.run == "T"

    labels = [marker.label for marker in recording.markers]
    assert labels == ["T0", "T1", "T0", "T2"]
    # idle[0, 2s) / 想象段 [提示+1s, 提示+4s)
    assert recording.markers[0].onset_sample == 0
    assert recording.markers[0].duration_samples == int(round(2.0 * RATE))
    assert recording.markers[1].duration_samples == int(round(3.0 * RATE))

    assert label_of_window(recording.markers, 0, int(2.0 * RATE)) == "idle"
    assert label_of_window(recording.markers, int(3.0 * RATE), int(5.0 * RATE)) == "left_hand"
    # 提示后第一秒是空的：不标任何类
    assert label_of_window(recording.markers, int(2.0 * RATE), int(3.0 * RATE)) is None


def test_feet_and_tongue_are_kept_but_unmapped() -> None:
    recording = _recording([3, 4])
    labels = [marker.label for marker in recording.markers]
    assert labels == ["T0", "T3", "T0", "T4"]
    mi = recording.markers[1]
    assert (
        label_of_window(recording.markers, mi.onset_sample, mi.onset_sample + int(2.0 * RATE))
        is None
    )


def test_units_are_converted_from_microvolt() -> None:
    payload = _run([1])
    payload["X"] = np.full_like(payload["X"], 1e6)  # 1e6 µV = 1 V
    recording = parse_bnci2014_mat([payload], subject=1, session="T")
    assert np.allclose(recording.samples, 1.0)


def test_sealed_session_requires_explicit_flag() -> None:
    missing = Path("_no_such_dir") / "A01E.mat"
    with pytest.raises(ContractError, match="封存测试 session"):
        load_bnci2014_session(missing, subject=1, session="E")
    with pytest.raises(ContractError, match="找不到 2a 文件"):
        load_bnci2014_session(missing, subject=1, session="E", allow_sealed=True)


def test_window_runs_are_grouped_by_run_not_trial() -> None:
    """两个 run 各 2 个 trial：窗的 run 号按 trial//48 分，绝不跨 run 混折。"""
    recording = parse_bnci2014_mat([_run([1, 2]), _run([1, 2])], subject=2, session="T")
    windows = [
        WindowRange(
            sample_rate_hz=RATE,
            start_sample=marker.onset_sample,
            end_sample=marker.onset_sample + int(2.0 * RATE),
        )
        for marker in recording.markers
        if marker.duration_samples is not None and marker.duration_samples >= int(2.0 * RATE)
    ]
    pairs = window_trial_and_run(recording, windows, trials_per_run=2)
    assert [run for _, run in pairs] == [0, 0, 0, 0, 0, 1, 1, 1]


def test_run_grouped_cv_produces_one_fold_per_run() -> None:
    config = AppConfig(source=replace(DEFAULT_CONFIG.source, sample_rate_hz=RATE))
    recording = parse_bnci2014_mat(
        [_run([1, 2, 3, 4, 1, 2]), _run([1, 2, 3, 4, 1, 2])],
        subject=3,
        session="T",
    )
    result = run_grouped_cv(recording, config, n_splits=2, trials_per_run=6)
    assert result["n_runs"] == 2
    assert len(result["folds"]) == 2
    assert set(result["windows_per_run"]) == {"0", "1"}
    for fold in result["folds"]:
        assert 0.0 <= fold["macro_f1"] <= 1.0
        assert fold["n_val"] > 0
