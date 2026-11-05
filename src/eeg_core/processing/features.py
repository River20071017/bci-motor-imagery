"""B 侧特征：每通道、每频带的对数方差，共 18 维。

与契约 ``ModelInputSpec.feature_layout = "channel_band"`` 对齐：

* 外层通道、内层频带，顺序固定为 ``MODEL_CHANNELS``；
* 每维是窗口内该通道该频带的方差取自然对数；
* 方差下限用 ``ProcessingConfig.log_variance_floor``（单位 V²），只防 0/负数；
* **不做**重参考、不再带通、不按窗标准化——这些属于 A 或模型包，不属于本函数。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from eeg_core.contracts import ContractError

__all__ = ["FEATURE_DIM", "log_variance_features"]

FEATURE_DIM = 18
"""9 个模型通道 × 2 个频带；首版固定。"""


def log_variance_features(
    band_8_13_hz: np.ndarray,
    band_13_30_hz: np.ndarray,
    channel_indices: Sequence[int],
    start_sample: int,
    end_sample: int,
    *,
    floor: float,
    ddof: int = 0,
) -> np.ndarray:
    """取一个窗口的 18 维特征。

    Parameters
    ----------
    band_8_13_hz, band_13_30_hz:
        ``FilterBank`` 输出的两路数组，形状 ``(通道数, 样本数)``，单位伏。
    channel_indices:
        9 个模型通道在数组中的行号，顺序即特征顺序。
    start_sample, end_sample:
        窗口 ``[start, end)``，必须落在数组内且非空。
    floor:
        方差下限，单位 V²；必须有限且 > 0。
    ddof:
        方差分母；训练与推理必须一致，默认 0（总体方差）。

    Returns
    -------
    numpy.ndarray
        形状 ``(len(channel_indices) * 2,)``；按 ``channel_band`` 先通道后频带。
    """
    first = np.asarray(band_8_13_hz)
    second = np.asarray(band_13_30_hz)
    if first.ndim != 2 or second.ndim != 2:
        raise ContractError(
            f"两路频带都必须是二维数组（通道 × 样本），收到 {first.ndim} 维 / {second.ndim} 维"
        )
    if first.shape != second.shape:
        raise ContractError(f"两路频带形状必须一致，收到 {first.shape} vs {second.shape}")
    if not np.issubdtype(first.dtype, np.floating) or not np.issubdtype(
        second.dtype, np.floating
    ):
        raise ContractError("两路频带必须是浮点数组（单位伏）")
    if isinstance(floor, bool) or not np.isfinite(floor) or floor <= 0:
        raise ContractError(f"floor 必须是有限正数，收到 {floor!r}")
    if ddof not in (0, 1):
        raise ContractError(f"ddof 只能是 0 或 1，收到 {ddof!r}")

    indices = np.asarray(list(channel_indices), dtype=np.int64)
    if indices.ndim != 1 or indices.size == 0:
        raise ContractError("channel_indices 必须是非空一维序列")
    if indices.min() < 0 or indices.max() >= first.shape[0]:
        raise ContractError(
            f"channel_indices 越界：数组只有 {first.shape[0]} 个通道，"
            f"收到 {indices.min()}..{indices.max()}"
        )
    n_samples = first.shape[1]
    start = int(start_sample)
    end = int(end_sample)
    if start < 0 or end > n_samples or start >= end:
        raise ContractError(f"窗口 [{start}, {end}) 不合法：数组共 {n_samples} 个样本")

    var_first = first[indices, start:end].var(axis=1, ddof=ddof)
    var_second = second[indices, start:end].var(axis=1, ddof=ddof)
    if not np.isfinite(var_first).all() or not np.isfinite(var_second).all():
        raise ContractError("窗口内存在非有限样本，无法计算方差；A 的质量层应已拦截")

    log_first = np.log(np.maximum(var_first, floor))
    log_second = np.log(np.maximum(var_second, floor))
    # np.stack(..., axis=1) 得到 [ch0 带1, ch0 带2, ch1 带1, ...]，再按 C 顺序拉平。
    return np.stack([log_first, log_second], axis=1).reshape(-1)
