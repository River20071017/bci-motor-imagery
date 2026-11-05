"""逐块质量检查；只观察模型通道的原始样本，不修改或清洗输入。"""

from __future__ import annotations

import math
import numbers

import numpy as np

from eeg_core.contracts import (
    DEFAULT_CONFIG,
    QUALITY_REASONS,
    AppConfig,
    ConfigError,
    ContractError,
    EEGChunk,
    QualityReport,
    ResetReason,
)

__all__ = ["QualityChecker"]


def _threshold(value: float, name: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ConfigError(f"{name} 必须是有限实数")
    number = float(value)
    if not math.isfinite(number) or number < 0 or (not allow_zero and number == 0):
        limit = "大于等于 0" if allow_zero else "大于 0"
        raise ConfigError(f"{name} 必须有限且{limit}")
    return number


class QualityChecker:
    """检查配置里的模型通道，并确认必需通道一个不缺。

    检查范围是 ``config.source.model_channels``（默认 9 个中央区通道），不是全部输入通道：
    重参考与特征都只用模型通道，其余通道的眼动/肌电伪迹不该把整块判成无效——否则干净的
    中央区数据会被挡在算法之外（S003/S009 实测）。非模型通道照原样随块传递，不检查。

    必须显式提供尚未冻结的三个质量阈值，不能把微伏直接当伏传入：

    * ``amplitude_limit_v``：伏；原始样本绝对值严格超过此上限即异常。
    * ``flatline_sec``：秒；连续常量段达到此时长即异常，折算为
      ``max(2, ceil(flatline_sec * sample_rate_hz))`` 个样本。
    * ``flatline_tolerance_v``：伏；相邻样本绝对差不超过此值视为未变化。
      0 表示严格相等；这是相邻差阈值，不是整段峰峰值阈值。

    常量段的末样本和连续计数跨块保留，块长不受限制。未声明间断时，
    seq 必须加一，样本区间必须首尾相接；采样率、通道表和类型不得变化。
    显式间断可开启新记录或向前跳过样本，但同记录不能重叠或跳 seq。
    首次调用（包括重置后）以当前块为起点，不要求从样本 0 开始。
    不合格后清空本检查器历史；下一块重新累计。
    """

    def __init__(
        self,
        *,
        amplitude_limit_v: float,
        flatline_sec: float,
        flatline_tolerance_v: float,
        config: AppConfig = DEFAULT_CONFIG,
    ) -> None:
        self.config = config
        self.model_channels = tuple(config.source.model_channels)
        self.amplitude_limit_v = _threshold(amplitude_limit_v, "amplitude_limit_v")
        self.flatline_sec = _threshold(flatline_sec, "flatline_sec")
        self.flatline_tolerance_v = _threshold(
            flatline_tolerance_v, "flatline_tolerance_v", allow_zero=True
        )
        self.reset()

    def reset(self, reason: ResetReason = ResetReason.SESSION_START) -> None:
        """幂等清理自身历史；暂停恢复保留位置与常量段计数。"""
        if reason == ResetReason.PAUSE_RESUME:
            return
        self._stream: tuple[object, ...] | None = None
        self._expected_seq: int | None = None
        self._expected_start: int | None = None
        self._last_values: np.ndarray | None = None
        self._constant_counts: np.ndarray | None = None

    def check(self, chunk: EEGChunk) -> QualityReport:
        """返回独立的块级报告；失败不抛质量异常，也不修补数据。"""
        if not isinstance(chunk, EEGChunk):
            raise ContractError("质量检查的输入必须是 EEGChunk")

        reasons: set[str] = set()
        bad_channels: set[str] = set()
        stream = (
            chunk.source_id,
            chunk.recording_id,
            chunk.sample_rate_hz,
            chunk.channel_names,
            chunk.channel_types,
        )
        if self._stream is not None:
            same_recording = stream[:2] == self._stream[:2]
            if same_recording:
                if (
                    chunk.seq != self._expected_seq
                    or chunk.start_sample < self._expected_start
                    or (not chunk.discontinuity and chunk.start_sample != self._expected_start)
                ):
                    reasons.add("sample_jump")
                if stream != self._stream:
                    reasons.add("interval_error")
            elif not chunk.discontinuity:
                reasons.add("sample_jump")

        if chunk.sample_rate_hz != self.config.source.sample_rate_hz:
            reasons.add("interval_error")

        missing = tuple(name for name in self.model_channels if name not in chunk.channel_names)
        if missing:
            reasons.add("missing_channel")
            bad_channels.update(missing)

        # 只对模型通道做样本级检查：重参考与特征都只消费它们，其余通道的伪迹不该拦下
        # 整块。缺失通道时只检查还在的模型通道，报告仍能说明哪些通道不可用。
        present = tuple(name for name in self.model_channels if name not in missing)
        model_samples = (
            chunk.samples[[chunk.channel_index(name) for name in present]] if present else None
        )
        if model_samples is not None:
            finite = np.isfinite(model_samples)
            non_finite = ~finite.all(axis=1)
            if non_finite.any():
                reasons.add("non_finite")
                bad_channels.update(
                    name for name, bad in zip(present, non_finite, strict=True) if bad
                )
            amplitude = (finite & (np.abs(model_samples) > self.amplitude_limit_v)).any(axis=1)
            if amplitude.any():
                reasons.add("amplitude")
                bad_channels.update(
                    name for name, bad in zip(present, amplitude, strict=True) if bad
                )

        # 不跨断点或不合格数据累计常量段，也不让 NaN/Inf 进入差分运算。
        if chunk.discontinuity:
            self._last_values = None
            self._constant_counts = None
        if not reasons:
            # reasons 为空说明模型通道一个不缺，model_samples 已经就位。
            assert model_samples is not None
            flatline = self._check_flatline(model_samples, chunk.sample_rate_hz)
            if flatline.any():
                reasons.add("flatline")
                bad_channels.update(
                    name for name, bad in zip(present, flatline, strict=True) if bad
                )

        if reasons:
            # 通道按输入顺序列出，缺失通道再按配置顺序追加。
            ordered_bad = tuple(
                name for name in (*chunk.channel_names, *missing) if name in bad_channels
            )
            report = QualityReport.failed(
                tuple(code for code in QUALITY_REASONS if code in reasons),
                bad_channels=ordered_bad,
            )
            self.reset(ResetReason.QUALITY_INVALID)
            return report

        self._stream = stream
        self._expected_seq = chunk.seq + 1
        self._expected_start = chunk.end_sample
        return QualityReport.ok()

    def _check_flatline(self, samples: np.ndarray, sample_rate_hz: float) -> np.ndarray:
        """只保留每通道末值和连续样本数，不缓存整段数据。

        ``samples`` 按 ``model_channels`` 的顺序排列，状态数组与它一一对应。
        """
        limit = max(2, math.ceil(self.flatline_sec * sample_rate_hz))
        if self._last_values is None:
            self._last_values = samples[:, 0].copy()
            self._constant_counts = np.zeros(samples.shape[0], dtype=np.int64)
        counts = self._constant_counts
        flatline = np.zeros(samples.shape[0], dtype=bool)
        for values in samples.T:
            unchanged = np.abs(values - self._last_values) <= self.flatline_tolerance_v
            counts = np.where(unchanged, np.minimum(counts + 1, limit), 1)
            flatline |= counts >= limit
            self._last_values = values.copy()
        self._constant_counts = counts
        return flatline
