"""契约 1.0 的事件：参考标注、重置原因、动作事件、运行时事件与会话状态。

三条硬规则：

1. **标签隔离**：参考标注（``ReferenceMarker``）只给参考轨和评估用，绝不放进 ``EEGChunk``，
   也不进解码路径（PRD §5.3、T03）。
2. **重置语义**：``ResetReason`` 说明「为什么必须忘掉历史」。按纪律，暂不为它抽象协议；
   运行时会话控制器和所有有状态的流式组件（滤波器、窗器、解码器）按本枚举实现即可。
   忘了重置的后果是隐蔽的：滤波器会把断点前后的样本算成一段连续信号，结果看起来正常但错得没边。
3. **两类事件分开**：``IntentEvent`` 是“模型认出了什么”（写 ``events.csv``）；
   ``RuntimeEvent`` 是“程序发生了什么”（写 ``runtime.jsonl``）。两者都用样本号/单调时钟，
   不混用墙钟。

``SessionState`` 是用户可见的运行状态机（PRD §6.4），与六种模型状态（``DecisionState``）
是两件事，必须分开保存。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from .frames import ContractError, _require_int, _require_real, _require_text
from .outputs import DecisionState

__all__ = [
    "END_REASONS",
    "IntentEvent",
    "ReferenceMarker",
    "ResetReason",
    "RuntimeEvent",
    "RuntimeEventKind",
    "SessionState",
]


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


class SessionState(StrEnum):
    """用户可见的运行状态机（PRD §6.4），与模型状态 ``DecisionState`` 分开保存。

    ``Ready → Running ⇄ Paused → Finished / Stopped / Error``；
    ``Finished`` / ``Stopped`` / ``Error`` 之后要重新开始，必须新建会话。
    """

    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    FINISHED = "finished"
    STOPPED = "stopped"
    ERROR = "error"


class RuntimeEventKind(StrEnum):
    """``runtime.jsonl`` 的事件类型；字符串枚举可直接写出，不必再维护映射。"""

    SESSION_START = "session_start"
    START = "start"
    PAUSE = "pause"
    RESUME = "resume"
    STOP = "stop"
    FINISH = "finish"
    DISCONTINUITY = "discontinuity"
    OVERLOAD = "overload"
    OVERRUN = "overrun"
    WRITE_ERROR = "write_error"
    ERROR = "error"


END_REASONS = (
    "state_changed",
    "paused",
    "stopped",
    "finished",
    "invalid",
    "expired",
)
"""动作事件的结束原因码：模型换类、用户暂停/停止、文件结束、质量异常、结果过期。"""


@dataclass(frozen=True, slots=True)
class RuntimeEvent:
    """运行中发生的一件事，写 ``runtime.jsonl``；与模型判断无关。

    ``at_monotonic_ms`` 是会话单调时钟（只算时间差），不是墙钟；``detail`` 给人看，
    可为空字符串，不参与任何判断。
    """

    session_id: str
    at_monotonic_ms: float
    kind: RuntimeEventKind
    detail: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "session_id", _require_text(self.session_id, "session_id"))
        object.__setattr__(
            self,
            "at_monotonic_ms",
            _require_real(self.at_monotonic_ms, "at_monotonic_ms", minimum=0.0),
        )
        if not isinstance(self.kind, RuntimeEventKind):
            raise ContractError(
                f"kind 必须是 RuntimeEventKind，收到 {type(self.kind).__name__}"
            )
        if not isinstance(self.detail, str):
            raise ContractError("detail 必须是字符串")
        object.__setattr__(self, "detail", self.detail.strip())

    def to_dict(self) -> dict[str, object]:
        """可直接写 JSONL 的普通字典。"""
        return {
            "session_id": self.session_id,
            "at_monotonic_ms": self.at_monotonic_ms,
            "kind": self.kind.value,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class IntentEvent:
    """一次连续同类动作的起止，写 ``events.csv``；只记录模型确认过的动作。

    时间用**样本号**表示，与 ``EEGChunk`` / ``WindowRange`` 同一口径；起止都带对应的
    决策序号（``DecisionFrame.seq``），D 据此回查证据窗。事件未结束时
    ``end_sample`` / ``end_decision_seq`` / ``end_reason`` 三者必须同时为 ``None``。
    """

    event_id: str
    session_id: str
    recording_id: str
    state: DecisionState
    start_sample: int
    start_decision_seq: int
    end_sample: int | None = None
    end_decision_seq: int | None = None
    end_reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_id", _require_text(self.event_id, "event_id"))
        object.__setattr__(self, "session_id", _require_text(self.session_id, "session_id"))
        object.__setattr__(
            self, "recording_id", _require_text(self.recording_id, "recording_id")
        )
        if not isinstance(self.state, DecisionState) or not self.state.is_action:
            raise ContractError("IntentEvent.state 只能是 left_hand 或 right_hand")
        object.__setattr__(
            self, "start_sample", _require_int(self.start_sample, "start_sample")
        )
        object.__setattr__(
            self, "start_decision_seq", _require_int(self.start_decision_seq, "start_decision_seq")
        )

        end_fields = (self.end_sample, self.end_decision_seq, self.end_reason)
        if all(value is None for value in end_fields):
            return
        if any(value is None for value in end_fields):
            raise ContractError(
                "事件结束时 end_sample / end_decision_seq / end_reason 必须同时给出"
            )

        end_sample = _require_int(self.end_sample, "end_sample")
        if end_sample <= self.start_sample:
            raise ContractError(
                f"end_sample 必须大于 start_sample，收到 {end_sample} <= {self.start_sample}"
            )
        end_seq = _require_int(self.end_decision_seq, "end_decision_seq")
        if end_seq < self.start_decision_seq:
            raise ContractError(
                "end_decision_seq 不能小于 start_decision_seq，"
                f"收到 {end_seq} < {self.start_decision_seq}"
            )
        reason = _require_text(self.end_reason, "end_reason")
        if reason not in END_REASONS:
            raise ContractError(f"end_reason 必须是 {END_REASONS} 之一，收到 {reason!r}")

        object.__setattr__(self, "end_sample", end_sample)
        object.__setattr__(self, "end_decision_seq", end_seq)
        object.__setattr__(self, "end_reason", reason)

    @property
    def is_open(self) -> bool:
        """动作是否还在进行。"""
        return self.end_sample is None

    @property
    def duration_samples(self) -> int | None:
        """持续样本数；未结束时为 ``None``（不猜）。"""
        if self.end_sample is None:
            return None
        return self.end_sample - self.start_sample

    def duration_sec(self, sample_rate_hz: float) -> float | None:
        """持续秒数，采样率由调用方提供；未结束时为 ``None``。"""
        samples = self.duration_samples
        if samples is None:
            return None
        return samples / _require_real(sample_rate_hz, "sample_rate_hz", strictly_positive=True)

    def to_dict(self) -> dict[str, object]:
        """可直接写 CSV 的普通字典；未结束字段为 ``None``。"""
        return {
            "event_id": self.event_id,
            "session_id": self.session_id,
            "recording_id": self.recording_id,
            "state": self.state.value,
            "start_sample": self.start_sample,
            "start_decision_seq": self.start_decision_seq,
            "end_sample": self.end_sample,
            "end_decision_seq": self.end_decision_seq,
            "end_reason": self.end_reason,
        }
