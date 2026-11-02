"""契约 1.0 的事件：输入侧的参考标注、运行时重置的原因码。

两条硬规则：

1. **标签隔离**：参考标注（``ReferenceMarker``）只给参考轨和评估用，绝不放进 ``EEGChunk``，
   也不进解码路径（PRD §5.3、T03）。
2. **重置语义**：``ResetReason`` 说明「为什么必须忘掉历史」。按纪律，暂不为它抽象协议；
   运行时会话控制器和所有有状态的流式组件（滤波器、窗器、解码器）按本枚举实现即可。
   忘了重置的后果是隐蔽的：滤波器会把断点前后的样本算成一段连续信号，结果看起来正常但错得没边。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from .frames import ContractError, _require_int, _require_text

__all__ = ["ReferenceMarker", "ResetReason"]


@dataclass(frozen=True, slots=True)
class ReferenceMarker:
    """一条原始标注：起止样本 + 原始标签。

    ``label`` 保留记录里的原样文字（如 PhysioNet 的 ``T0``/``T1``/``T2``）。映射成
    idle / left_hand / right_hand 属于评估口径（D 负责），不在这里做。
    时间只用样本编号表达，与 ``EEGChunk`` 同一口径。
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
            raise ContractError(f"sample_rate_hz 必须大于 0，收到 {sample_rate_hz!r}")
        return self.onset_sample / rate

    def covers_sample(self, sample: int) -> bool | None:
        """判断某样本是否落在标注区间内；无时长时返回 ``None``（未知，不当成覆盖）。"""
        if self.duration_samples is None:
            return None
        return self.onset_sample <= sample < self.onset_sample + self.duration_samples


class ResetReason(StrEnum):
    """请求重置的原因；字符串枚举可直接写进 ``runtime.jsonl``，不必再维护映射。

    各原因要求的效果（实现方按此清理，没有协议接口也照此执行）：

    ``SESSION_START`` / ``DISCONTINUITY`` / ``QUALITY_INVALID``
        丢弃全部与样本有关的状态——滤波状态（``zi``）、未完成窗口、预热计数、候选确认、
        当前动作的连续判定；在下一次预热满足前，输出只能是 ``warming_up`` 或 ``invalid``。
    ``PAUSE_RESUME``
        保留滤波状态与已处理历史（数据没断），只作废候选确认，避免把暂停前已发布的窗
        重复发布。

    ``reset`` 必须幂等，且不抛异常干扰会话控制。
    """

    SESSION_START = "session_start"
    DISCONTINUITY = "discontinuity"
    QUALITY_INVALID = "quality_invalid"
    PAUSE_RESUME = "pause_resume"
