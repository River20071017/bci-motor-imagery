"""界面测试用模拟 EEG 数据源（非真实采集 / 算法输出）。

分工方案 4.2 允许：前期 C 可使用明确标识为"界面测试"的模拟结果对象，
D 可使用格式样例并行开发；10月29日起集成演示应接真实处理结果。

本模块实现团队契约 ``eeg_core.contracts.DataSource`` 协议：
- ``info`` 返回 ``SourceInfo``（首次读取即可得到记录元数据）。
- ``read_chunk()`` 返回下一块 ``EEGChunk``；有限源结束后返回 ``None``。
- ``close()`` / 上下文管理器释放资源。

产生的 ``EEGChunk`` 字段与团队契约完全一致（单位伏、只读 float64、首块标 discontinuity），
因此后续接入 A 的真实数据源时，C 的界面与回放控制器无需改动。
"""
from __future__ import annotations

import time

import numpy as np

from eeg_core.contracts import EEGChunk, SourceInfo

# PRD 默认显示的 8 个通道（FC3、FCz、FC4、C3、Cz、C4、CP3、CPz）
DEFAULT_CHANNELS = ["FC3", "FCz", "FC4", "C3", "Cz", "C4", "CP3", "CPz"]


class MockEegSource:
    """产生类似多通道脑电的连续信号块（单位：伏），实现 ``DataSource`` 协议。"""

    def __init__(
        self,
        sample_rate_hz: float = 160.0,
        channels: list | None = None,
        chunk_samples: int = 8,
        source_id: str = "MOCK",
        recording_id: str = "S001R12",
        seed: int = 20261022,
        max_chunks: int | None = None,
    ) -> None:
        self.sample_rate_hz = sample_rate_hz
        self.channels = list(channels) if channels else list(DEFAULT_CHANNELS)
        self.chunk_samples = chunk_samples
        self.source_id = source_id
        self.recording_id = recording_id
        self.max_chunks = max_chunks
        self.n_channels = len(self.channels)
        self._rng = np.random.default_rng(seed)
        self._info = SourceInfo(
            source_id=source_id,
            recording_id=recording_id,
            sample_rate_hz=sample_rate_hz,
            channel_names=tuple(self.channels),
            reference=None,
            subject_id=None,
            run=None,
            n_samples=None,
        )
        self._t0 = time.perf_counter()
        self.reset()

    # ---- DataSource 协议 ----
    @property
    def info(self) -> SourceInfo:
        return self._info

    def read_chunk(self) -> EEGChunk | None:
        if self.max_chunks is not None and self._seq >= self.max_chunks:
            return None
        n = self.chunk_samples
        t = np.arange(self._start_sample, self._start_sample + n) / self.sample_rate_hz
        # 基础噪声（约 5 µV）
        sig = self._rng.standard_normal((self.n_channels, n)) * 5e-6
        # 每通道不同频率的慢漂移，使其看起来像 EEG（约 8~30 Hz 频段附近）
        for i in range(self.n_channels):
            freq = 9.0 + i * 1.3
            sig[i] += 8e-6 * np.sin(2.0 * np.pi * freq * t + i)
        discontinuity = (self._seq == 0)  # 首块标新记录边界
        chunk = EEGChunk(
            source_id=self.source_id,
            recording_id=self.recording_id,
            seq=self._seq,
            start_sample=self._start_sample,
            sample_rate_hz=self.sample_rate_hz,
            channel_names=tuple(self.channels),
            samples=sig,  # 伏
            released_monotonic_ms=(time.perf_counter() - self._t0) * 1000.0,
            discontinuity=discontinuity,
        )
        self._seq += 1
        self._start_sample += n
        return chunk

    def close(self) -> None:
        self._closed = True

    def reset(self) -> None:
        self._seq = 0
        self._start_sample = 0
        self._closed = False

    def __enter__(self) -> "MockEegSource":
        self.reset()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
