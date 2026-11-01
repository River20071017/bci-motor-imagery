"""原始记录自带的标注事件：参考轨与评估专用，**不进解码路径**。

PRD §5.3 的标签隔离是硬规则：在线解码路径只接收 EEG 数据块与采样元数据，不接收类别
标签、标注时间表或试次编号。所以原始标注不能塞进 ``EEGChunk``，而是单独成对象，由
:class:`~eeg_core.contracts.protocols.MarkerSource` 提供给参考轨与评估代码。

本模块描述的是**输入侧的参考标注**；模型输出侧的动作事件（``IntentEvent``）是另一回事，
后续单独实现，两者不要混用。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .frames import EEGChunkError, _require_int, _require_text

__all__ = ["ReferenceMarker"]


@dataclass(frozen=True, slots=True)
class ReferenceMarker:
    """一条原始标注：起止样本 + 原始标签。

    ``label`` 保留记录里的原样文字（如 PhysioNet 的 ``T0``/``T1``/``T2``）。把它映射成
    idle / left_hand / right_hand 属于评估口径（D 负责），不在这里做——否则「标签含义」
    会提前固化进在线路径。

    时间只用样本编号表达，与 ``EEGChunk`` 同一口径，避免出现两套时钟。
    """

    recording_id: str
    onset_sample: int
    label: str
    duration_samples: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "recording_id", _require_text(self.recording_id, "recording_id"))
        object.__setattr__(self, "onset_sample", _require_int(self.onset_sample, "onset_sample"))
        object.__setattr__(self, "label", _require_text(self.label, "label"))
        if self.duration_samples is not None:
            object.__setattr__(
                self,
                "duration_samples",
                _require_int(self.duration_samples, "duration_samples"),
            )

    @property
    def end_sample(self) -> int | None:
        """标注结束样本（左闭右开）；记录没给时长时为 ``None``——不猜。"""
        if self.duration_samples is None:
            return None
        return self.onset_sample + self.duration_samples

    def onset_t_sec(self, sample_rate_hz: float) -> float:
        """标注起点的源时间（秒）；采样率由调用方提供，标注本身不复制一份。"""
        rate = float(sample_rate_hz)
        if not math.isfinite(rate) or rate <= 0.0:
            raise EEGChunkError(f"sample_rate_hz 必须大于 0，收到 {sample_rate_hz!r}")
        return self.onset_sample / rate

    def covers_sample(self, sample: int) -> bool | None:
        """判断某样本是否落在标注区间内；无时长时返回 ``None``（未知，不当成覆盖）。"""
        if self.duration_samples is None:
            return None
        return self.onset_sample <= sample < self.onset_sample + self.duration_samples
