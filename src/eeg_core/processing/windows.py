"""B 侧分窗：只负责窗口网格与「整窗可用」判断，不碰滤波/质量。

窗口规则来自契约与 A 的预热血统：

* 2 s 窗 = 320 样本，250 ms 步长 = 40 样本，网格锚定记录的样本 0；
* 整窗必须落在同一段连续有效数据里；
* 窗起点距该段起点至少 3 s（480 样本），与 ``FilterBank`` 的预热阈值一致；
* 不跨 invalid / discontinuity，宁可不产窗。

``FilterBank`` 返回的 ``ready`` 是「块末是否已预热」，块内前几个样本可能还没满 3 s；
所以这里不用 ``ready`` 逐样本标记，而是用「段起点 + warmup」判断窗起点，保守且与
``WindowRange`` 的文档一致。
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import numpy as np

from eeg_core.contracts import AppConfig, ContractError, WindowRange

__all__ = [
    "hop_samples",
    "is_window_usable",
    "iter_window_ranges",
    "required_warmup_samples",
    "window_samples",
]


def window_samples(config: AppConfig) -> int:
    """一个观察窗的样本数（160 Hz × 2 s = 320）。"""
    return max(
        1,
        int(round(config.processing.window_sec * config.source.sample_rate_hz)),
    )


def hop_samples(config: AppConfig) -> int:
    """相邻窗起点的样本间隔（160 Hz × 250 ms = 40）。"""
    return max(
        1,
        int(round(config.processing.step_sec * config.source.sample_rate_hz)),
    )


def required_warmup_samples(config: AppConfig) -> int:
    """与 ``FilterBank`` 一致的预热门槛：max(3 s, warmup_sec) 折算成样本。"""
    seconds = max(3.0, config.processing.warmup_sec)
    return max(1, int(math.ceil(seconds * config.source.sample_rate_hz)))


def iter_window_ranges(
    n_samples: int,
    config: AppConfig,
    *,
    anchor_sample: int = 0,
) -> Iterator[WindowRange]:
    """按固定网格产出完整窗口 ``[start, end)``。

    只产完整窗；最后一个不足一窗的尾巴不补、不猜。``anchor_sample`` 默认 0，
    即记录的第一个样本。
    """
    if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples < 0:
        raise ContractError(f"n_samples 必须是 >= 0 的整数，收到 {n_samples!r}")
    window = window_samples(config)
    hop = hop_samples(config)
    if n_samples < window:
        return
    for start in range(anchor_sample, n_samples - window + 1, hop):
        yield WindowRange(
            sample_rate_hz=config.source.sample_rate_hz,
            start_sample=start,
            end_sample=start + window,
        )


def is_window_usable(
    window: WindowRange,
    valid: np.ndarray,
    segment_start: np.ndarray,
    *,
    warmup: int,
) -> bool:
    """整窗是否可用：全样本有效、同一连续段、且起点已过预热。

    ``valid[i]`` 表示第 i 个样本是否来自一次有效块；无效块（A 判 ``invalid``）
    及其后的预热样本都应为 ``False``。``segment_start[i]`` 是该样本所在连续段的
    起始样本号，无效样本填 ``-1``；段在 invalid / ``discontinuity`` 处重新开始。
    """
    if not isinstance(window, WindowRange):
        raise ContractError("window 必须是 WindowRange")
    if warmup < 0:
        raise ContractError(f"warmup 不能为负，收到 {warmup}")

    valid_array = np.asarray(valid)
    segment_array = np.asarray(segment_start)
    if valid_array.ndim != 1 or segment_array.ndim != 1:
        raise ContractError("valid / segment_start 必须是一维数组")
    if valid_array.shape != segment_array.shape:
        raise ContractError(
            "valid 与 segment_start 长度必须一致，"
            f"收到 {valid_array.shape} vs {segment_array.shape}"
        )

    start, end = window.start_sample, window.end_sample
    if start < 0 or end > valid_array.shape[0]:
        return False
    if not bool(valid_array[start:end].all()):
        return False

    segment = int(segment_array[start])
    if segment < 0:
        return False
    if segment != int(segment_array[end - 1]):
        return False
    return start >= segment + warmup
