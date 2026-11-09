"""契约 1.0 的输出侧数据：质量结论、决策状态、三状态分数、逐窗结果。

这些都从处理链「出来」，只给界面和存档用。和输入侧一样全部 ``frozen``：
谁也不能在流水线中途偷偷改一条已经发布的结果。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum

from .frames import (
    CONTRACT_VERSION,
    ContractError,
    WindowRange,
    _require_int,
    _require_real,
    _require_text,
)

QUALITY_REASONS = (
    "non_finite",
    "missing_channel",
    "sample_jump",
    "interval_error",
    "flatline",
    "amplitude",
)
"""质量原因码：固定集合。界面、CSV 与测试都按这些字符串断言，不做同义词。"""

__all__ = [
    "QUALITY_REASONS",
    "QualityReport",
    "DecisionState",
    "Scores",
    "DecisionFrame",
    "utc_now_iso",
]

_UTC_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$")
"""``generated_at_utc`` 的格式：``2026-10-16T07:00:12.080Z``（UTC，毫秒可选）。"""


def utc_now_iso() -> str:
    """当前 UTC 时间，按契约格式返回；统一入口，避免各模块自己拼时间串。"""
    moment = datetime.now(UTC)
    return moment.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


@dataclass(frozen=True, slots=True)
class QualityReport:
    """一次质量检查的结论：能不能用、为什么不能用、哪些通道有问题。

    ``reasons`` 只能取 :data:`QUALITY_REASONS` 里的原因码，保证界面、CSV 与测试说的是
    同一件事。``valid`` 与 ``reasons`` 互斥：合格不许带原因，不合格必须给原因。
    ``window`` 用于窗口级检查；块级快速检查（非有限值、缺通道）没有窗口，留 ``None``。
    """

    valid: bool
    reasons: tuple[str, ...] = ()
    bad_channels: tuple[str, ...] = ()
    window: WindowRange | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.valid, bool):
            raise ContractError("valid 必须是 bool")
        if self.reasons is None or isinstance(self.reasons, (str, bytes)):
            raise ContractError("reasons 必须是原因码序列（不能为 None 或单个字符串）")
        if self.bad_channels is None or isinstance(self.bad_channels, (str, bytes)):
            raise ContractError("bad_channels 必须是通道名序列（不能为 None 或单个字符串）")
        if self.window is not None and not isinstance(self.window, WindowRange):
            raise ContractError("window 必须是 WindowRange 或 None")

        reasons: list[str] = []
        for index, item in enumerate(tuple(self.reasons)):
            if not isinstance(item, str) or item not in QUALITY_REASONS:
                raise ContractError(
                    f"reasons[{index}] 必须是 {QUALITY_REASONS} 之一，收到 {item!r}"
                )
            if item in reasons:
                raise ContractError(f"reasons 重复：{item}")
            reasons.append(item)

        channels: list[str] = []
        for index, item in enumerate(tuple(self.bad_channels)):
            if not isinstance(item, str) or not item.strip():
                raise ContractError(f"bad_channels[{index}] 必须是非空字符串")
            name = item.strip()
            if name in channels:
                raise ContractError(f"bad_channels 重复：{name}")
            channels.append(name)

        if self.valid and (reasons or channels):
            raise ContractError("valid=True 时不能带 reasons 或 bad_channels")
        if not self.valid and not reasons:
            raise ContractError("valid=False 时必须给出至少一个原因码")

        # 原因码按 QUALITY_REASONS 的固定顺序排列，保证 CSV/日志在不同实现下可比。
        reasons = [code for code in QUALITY_REASONS if code in set(reasons)]

        object.__setattr__(self, "reasons", tuple(reasons))
        object.__setattr__(self, "bad_channels", tuple(channels))

    @classmethod
    def ok(cls, window: WindowRange | None = None) -> QualityReport:
        """合格报告。"""
        return cls(valid=True, window=window)

    @classmethod
    def failed(
        cls,
        reasons: tuple[str, ...] | list[str],
        *,
        bad_channels: tuple[str, ...] | list[str] = (),
        window: WindowRange | None = None,
    ) -> QualityReport:
        """不合格报告；``reasons`` 至少一个原因码。"""
        return cls(valid=False, reasons=reasons, bad_channels=bad_channels, window=window)


class DecisionState(StrEnum):
    """逐窗决策的六种状态（PRD §6.3）；运行状态（Ready/Running/…）是另一回事。

    ``warming_up`` 历史不足、``invalid`` 数据不可用、``uncertain`` 拒判；
    **低分不能直接转成 idle**。只有左右手会驱动动画。
    """

    WARMING_UP = "warming_up"
    IDLE = "idle"
    LEFT_HAND = "left_hand"
    RIGHT_HAND = "right_hand"
    UNCERTAIN = "uncertain"
    INVALID = "invalid"

    @property
    def is_action(self) -> bool:
        """是否为会驱动动画的动作状态（左手或右手）。"""
        return self in (DecisionState.LEFT_HAND, DecisionState.RIGHT_HAND)


@dataclass(frozen=True, slots=True)
class Scores:
    """三状态分数：0~1、有限；**不要求相加等于 1**（未校准概率，不能当置信度）。"""

    idle: float
    left_hand: float
    right_hand: float

    def __post_init__(self) -> None:
        for name in ("idle", "left_hand", "right_hand"):
            value = _require_real(getattr(self, name), name, minimum=0.0)
            if value > 1.0:
                raise ContractError(f"{name} 不能大于 1，收到 {value!r}")
            object.__setattr__(self, name, value)

    def as_dict(self) -> dict[str, float]:
        """按 idle、left_hand、right_hand 的顺序返回字典。"""
        return {"idle": self.idle, "left_hand": self.left_hand, "right_hand": self.right_hand}

    def top(self) -> tuple[str, float]:
        """最高分类别与分值；并列时按字典顺序取第一个。"""
        return max(self.as_dict().items(), key=lambda item: item[1])


@dataclass(frozen=True, slots=True)
class DecisionFrame:
    """一个更新点的决策结果，字段与 PRD §7.2 的 JSON 示例一一对应。

    一次决策 = 证据窗 + 质量结论 + 状态 +（可选）分数 + 溯源信息。

    校验规则（构造时强制）：

    * ``warming_up``：没有窗口、没有分数、没有 intent；``invalid``：质量必须不合格，
      窗口可有（窗内质量失败）可无（流中断）。
    * ``idle`` / ``uncertain`` / ``left_hand`` / ``right_hand``：必须有窗口和分数，质量合格。
    * ``intent`` 只能出现且等于 ``left_hand`` / ``right_hand``；其他状态一律 ``None``。
    * 有窗口时，``sample_rate_hz`` 与 ``source_t_sec`` 必须和窗口一致（末端边界时间）。
    * ``generated_at_utc`` 是 UTC 的 ISO 时间；``emitted_monotonic_ms`` 是会话单调时钟，
      两者不能混用。写出的 JSON 不允许出现 NaN / Infinity。
    """

    session_id: str
    recording_id: str
    seq: int
    decision_state: DecisionState
    sample_rate_hz: float
    source_t_sec: float
    generated_at_utc: str
    emitted_monotonic_ms: float
    model_id: str
    pipeline_version: str
    config_hash: str
    window: WindowRange | None = None
    intent: DecisionState | None = None
    scores: Scores | None = None
    score_type: str = "uncalibrated_probability"
    quality: QualityReport = field(default_factory=QualityReport.ok)
    valid_for_ms: int = 750
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        version = _require_text(self.schema_version, "schema_version")
        if version != CONTRACT_VERSION:
            raise ContractError(
                f"不支持的 schema_version {version!r}，当前契约版本为 {CONTRACT_VERSION}"
            )
        object.__setattr__(self, "schema_version", version)

        for name in ("session_id", "recording_id", "model_id", "pipeline_version", "config_hash"):
            object.__setattr__(self, name, _require_text(getattr(self, name), name))
        object.__setattr__(self, "seq", _require_int(self.seq, "seq"))
        object.__setattr__(
            self,
            "sample_rate_hz",
            _require_real(self.sample_rate_hz, "sample_rate_hz", strictly_positive=True),
        )
        object.__setattr__(
            self,
            "source_t_sec",
            _require_real(self.source_t_sec, "source_t_sec", minimum=0.0),
        )
        generated = _require_text(self.generated_at_utc, "generated_at_utc")
        if _UTC_ISO_RE.match(generated) is None:
            raise ContractError(
                "generated_at_utc 必须是 UTC 的 ISO 时间"
                f"（如 2026-10-16T07:00:12.080Z），收到 {generated!r}"
            )
        object.__setattr__(self, "generated_at_utc", generated)
        object.__setattr__(
            self,
            "emitted_monotonic_ms",
            _require_real(self.emitted_monotonic_ms, "emitted_monotonic_ms", minimum=0.0),
        )
        object.__setattr__(
            self,
            "valid_for_ms",
            _require_int(self.valid_for_ms, "valid_for_ms", minimum=1),
        )
        object.__setattr__(self, "score_type", _require_text(self.score_type, "score_type"))

        if not isinstance(self.decision_state, DecisionState):
            raise ContractError(
                f"decision_state 必须是 DecisionState，收到 {type(self.decision_state).__name__}"
            )
        if self.scores is not None and not isinstance(self.scores, Scores):
            raise ContractError("scores 必须是 Scores 或 None")
        if not isinstance(self.quality, QualityReport):
            raise ContractError("quality 必须是 QualityReport")
        if self.intent is not None and not isinstance(self.intent, DecisionState):
            raise ContractError("intent 必须是 DecisionState 或 None")
        if self.window is not None and not isinstance(self.window, WindowRange):
            raise ContractError("window 必须是 WindowRange 或 None")

        state = self.decision_state
        self._check_state_rules(state)
        self._check_window(state)

    def _check_state_rules(self, state: DecisionState) -> None:
        if self.intent is not None:
            if not self.intent.is_action:
                raise ContractError(
                    f"intent 只能是 left_hand 或 right_hand，收到 {self.intent.value!r}"
                )
            if self.intent is not state:
                raise ContractError(
                    f"intent 与 decision_state 不一致：{self.intent.value} vs {state.value}"
                )
        elif state.is_action:
            raise ContractError(f"{state.value} 状态必须给出 intent")

        if state.is_action or state in (DecisionState.IDLE, DecisionState.UNCERTAIN):
            if self.scores is None:
                raise ContractError(f"{state.value} 状态必须有模型分数")
        elif self.scores is not None:
            raise ContractError(f"{state.value} 状态不应带模型分数")

        if state is DecisionState.INVALID:
            if self.quality.valid:
                raise ContractError("invalid 状态必须带不合格的质量报告")
        elif not self.quality.valid:
            raise ContractError("只有 invalid 状态可以带不合格的质量报告")

    def _check_window(self, state: DecisionState) -> None:
        if state is DecisionState.WARMING_UP:
            if self.window is not None:
                raise ContractError("warming_up 状态不应带完整窗口")
            return
        if self.window is None:
            if state is not DecisionState.INVALID:
                raise ContractError(f"{state.value} 状态必须带证据窗")
            return
        if abs(self.window.sample_rate_hz - self.sample_rate_hz) > 1e-9:
            raise ContractError("window 的采样率与 sample_rate_hz 不一致")
        if abs(self.window.source_t_sec - self.source_t_sec) > 1e-9:
            raise ContractError(
                f"source_t_sec 必须等于窗口末端时间 {self.window.source_t_sec:g}，"
                f"收到 {self.source_t_sec:g}"
            )

    @property
    def decision_id(self) -> str:
        """会话内唯一的决策标识，如 ``demo-20261016-001/000048``。"""
        return f"{self.session_id}/{self.seq:06d}"

    @property
    def is_action(self) -> bool:
        """是否是可以驱动动画的动作状态。"""
        return self.decision_state.is_action

    def to_dict(self) -> dict[str, object]:
        """按 PRD §7.2 的字段名输出；窗口缺失时两个样本字段为 ``None``。"""
        window = self.window
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "recording_id": self.recording_id,
            "decision_id": self.decision_id,
            "seq": self.seq,
            "window_start_sample": window.start_sample if window is not None else None,
            "window_end_sample": window.end_sample if window is not None else None,
            "sample_rate_hz": self.sample_rate_hz,
            "source_t_sec": self.source_t_sec,
            "generated_at_utc": self.generated_at_utc,
            "emitted_monotonic_ms": self.emitted_monotonic_ms,
            "decision_state": self.decision_state.value,
            "intent": self.intent.value if self.intent is not None else None,
            "scores": self.scores.as_dict() if self.scores is not None else None,
            "score_type": self.score_type,
            "quality": {
                "valid": self.quality.valid,
                "reasons": list(self.quality.reasons),
                "bad_channels": list(self.quality.bad_channels),
            },
            "model_id": self.model_id,
            "pipeline_version": self.pipeline_version,
            "config_hash": self.config_hash,
            "valid_for_ms": self.valid_for_ms,
        }

    def to_json(self) -> str:
        """一行 JSON，可直接写 ``decisions.jsonl``；NaN / Infinity 会被拒绝写出。"""
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
