"""质量检查 → 模型通道平均参考 → 两路有状态 SOS 因果滤波。"""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import butter, sosfilt

from eeg_core.contracts import (
    DEFAULT_CONFIG,
    AppConfig,
    ConfigError,
    EEGChunk,
    ResetReason,
)

from .quality import QualityChecker

__all__ = ["FilterBank"]


class FilterBank:
    """最小流式处理入口；每条流使用一个实例，按源样本顺序调用 process。

    滤波参数读取 ``config.processing``，采样率、参考通道读取 ``config.source``。
    本入口支持配置约定的 8–13 / 13–30 Hz 两路 Butterworth SOS；默认四阶。
    ``amplitude_limit_v``、``flatline_sec``、``flatline_tolerance_v`` 的单位和
    语义见 :class:`QualityChecker`，三个参数均必须显式提供。

    模型通道的逐样本均值从所有输入行扣除；输出保留全部输入通道的顺序和伏单位。
    使用零初始状态的 ``sosfilt``，每频带每通道独立保存末状态，无双向滤波或未来样本。
    每段有效数据至少累计 max(3 秒, 配置预热时长)，才标记 ready。
    """

    def __init__(
        self,
        *,
        amplitude_limit_v: float,
        flatline_sec: float,
        flatline_tolerance_v: float,
        config: AppConfig = DEFAULT_CONFIG,
    ) -> None:
        source = config.source
        processing = config.processing
        if source.reference_policy != "model_channel_average":
            raise ConfigError("FilterBank 仅支持 model_channel_average 重参考")
        if processing.filter_impl != "butterworth_sos":
            raise ConfigError("FilterBank 仅支持 butterworth_sos 因果滤波")
        if processing.bands_hz != ((8.0, 13.0), (13.0, 30.0)):
            raise ConfigError("FilterBank 要求频带为 8–13 Hz 和 13–30 Hz")
        if source.sample_rate_hz / 2 <= max(high for _, high in processing.bands_hz):
            raise ConfigError("采样率的 Nyquist 频率必须高于所有频带上限")

        self.config = config
        self._quality = QualityChecker(
            amplitude_limit_v=amplitude_limit_v,
            flatline_sec=flatline_sec,
            flatline_tolerance_v=flatline_tolerance_v,
            config=config,
        )
        self._sos = tuple(
            butter(
                processing.filter_order,
                band,
                btype="bandpass",
                fs=source.sample_rate_hz,
                output="sos",
            )
            for band in processing.bands_hz
        )
        self._warmup_samples = math.ceil(
            max(3.0, processing.warmup_sec) * source.sample_rate_hz
        )
        self.reset()

    def reset(self, reason: ResetReason = ResetReason.SESSION_START) -> None:
        """仅重置本模块；PAUSE_RESUME 完整保留滤波、质量历史和预热计数。"""
        if reason == ResetReason.PAUSE_RESUME:
            return
        self._quality.reset(reason)
        self._reset_filters()

    def _reset_filters(self) -> None:
        self._zi: list[np.ndarray] | None = None
        self._valid_samples = 0

    def process(self, chunk: EEGChunk) -> dict[str, object]:
        """处理当前块，原始 EEGChunk 及其样本始终不变。

        字典保留 ``chunk.metadata()`` 的字段，并补充 chunk_id、n_samples、
        end_sample、unit，以及以下处理结果：

        * ``quality``：现有 QualityReport；块级 window 为 None。
        * ``band_8_13_hz`` / ``band_13_30_hz``：独立的 float64 数组，
          形状与输入 samples 相同；invalid 时均为 None。
        * ``ready``：在当前块末端是否完成预热。
        * ``status``：invalid / warming_up / ready（处理状态，不是解码决策）。

        预热期间仍返回因果滤波数组，但 ready=False；无效块不会进入滤波器。
        显式 discontinuity 的有效块作为新段起点参与重新预热。
        """
        quality = self._quality.check(chunk)
        result: dict[str, object] = {
            **chunk.metadata(),
            "chunk_id": chunk.chunk_id,
            "n_samples": chunk.n_samples,
            "end_sample": chunk.end_sample,
            "unit": chunk.unit,
            "quality": quality,
            "band_8_13_hz": None,
            "band_13_30_hz": None,
            "ready": False,
            "status": "invalid",
        }
        if not quality.valid:
            self.reset(ResetReason.QUALITY_INVALID)
            return result

        if chunk.discontinuity:
            self._reset_filters()
        model_indices = [
            chunk.channel_index(name) for name in self.config.source.model_channels
        ]
        reference = chunk.samples[model_indices].mean(axis=0, keepdims=True)
        referenced = chunk.samples - reference

        if self._zi is None:
            # axis=-1 对应样本轴，zi 维度为 (SOS 节数, 通道数, 2)。
            self._zi = [
                np.zeros((len(sos), chunk.n_channels, 2), dtype=np.float64)
                for sos in self._sos
            ]
        for index, key in enumerate(("band_8_13_hz", "band_13_30_hz")):
            filtered, self._zi[index] = sosfilt(
                self._sos[index], referenced, axis=-1, zi=self._zi[index]
            )
            result[key] = filtered

        self._valid_samples += chunk.n_samples
        ready = self._valid_samples >= self._warmup_samples
        result["ready"] = ready
        result["status"] = "ready" if ready else "warming_up"
        return result
