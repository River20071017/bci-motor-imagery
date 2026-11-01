"""契约 1.0 的统一输入块 :class:`EEGChunk`（PRD §7.1）。

在线路径（sources → runtime → processing → decoding → ui / outputs）只通过本结构传递
脑电样本，因此这里同时锁死三件事：

1. **单位**：``samples`` 一律为伏（:data:`EEG_UNIT`）。EDF 适配器负责把记录单位换算成
   伏，绘图层再乘 ``1e6`` 显示为微伏；中间任何一层都不再换算。
2. **时间**：只由样本编号与采样率推导（``start_sample / sample_rate_hz``）。于是同一段
   样本按不同块长供给时，窗口边界完全一致（验收 T05），也不依赖墙钟或界面刷新。
3. **标签隔离**：结构里没有类别、试次编号或参考标注时间表。参考轨只能走独立对象与独立
   时钟（PRD §5.3）。

约定：

* 通道顺序即 ``samples`` 的行顺序；契约不排序、不改名、不增删通道。
* 构造时把数组拷贝成只读、C 连续的 ``float64``。块很小（160 Hz 下约 8 个样本），拷贝换来
  「发布后不会被上游改写」的保证：上游改自己的缓冲区不影响已发出的块。
* 非有限值（NaN / Inf）原样放行：T07 要让质量层看到真实异常并判 ``invalid``。契约不做静默
  清洗，也不把坏数据伪装成好数据。
"""

from __future__ import annotations

import json
import math
import numbers
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum

import numpy as np

CONTRACT_VERSION = "1.0"
"""当前输入契约版本；字段或语义变化必须同时升级版本并更新 docs/01。"""

EEG_UNIT = "V"
"""``EEGChunk.samples`` 的内部单位，固定为伏。"""

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
    "CONTRACT_VERSION",
    "EEG_UNIT",
    "QUALITY_REASONS",
    "DecisionFrame",
    "DecisionState",
    "EEGChunk",
    "EEGChunkError",
    "QualityReport",
    "Scores",
    "WindowRange",
    "utc_now_iso",
]

_UTC_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$")
"""``generated_at_utc`` 的格式：``2026-10-16T07:00:12.080Z``（UTC，毫秒可选）。"""


def utc_now_iso() -> str:
    """当前 UTC 时间，按契约格式返回；统一入口，避免各模块自己拼时间串。"""
    moment = datetime.now(timezone.utc)
    return moment.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


class EEGChunkError(ValueError):
    """数据块或元数据不满足契约。

    适配器应当把这一异常当作「输入不可用」上报（T02：阻止运行并说明原因），
    而不是猜测单位、通道顺序或采样率去修正。
    """


def _require_text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise EEGChunkError(f"{field_name} 必须是字符串，收到 {type(value).__name__}")
    text = value.strip()
    if not text:
        raise EEGChunkError(f"{field_name} 不能为空")
    return text


def _require_int(value: object, field_name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise EEGChunkError(f"{field_name} 必须是整数，收到 {type(value).__name__}")
    number = int(value)
    if number < minimum:
        raise EEGChunkError(f"{field_name} 不能小于 {minimum}，收到 {number}")
    return number


def _require_real(
    value: object,
    field_name: str,
    *,
    minimum: float | None = None,
    strictly_positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise EEGChunkError(f"{field_name} 必须是实数，收到 {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise EEGChunkError(f"{field_name} 必须是有限值，收到 {number!r}")
    if strictly_positive and number <= 0.0:
        raise EEGChunkError(f"{field_name} 必须大于 0，收到 {number!r}")
    if minimum is not None and number < minimum:
        raise EEGChunkError(f"{field_name} 不能小于 {minimum}，收到 {number!r}")
    return number


def _require_channel_names(values: object) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise EEGChunkError("channel_names 必须是通道名序列，不能是单个字符串")
    try:
        items = list(values)  # type: ignore[arg-type]
    except TypeError:
        raise EEGChunkError("channel_names 必须是字符串序列") from None
    if not items:
        raise EEGChunkError("channel_names 不能为空")

    names: list[str] = []
    for index, item in enumerate(items):
        if not isinstance(item, str):
            raise EEGChunkError(f"channel_names[{index}] 必须是字符串")
        name = item.strip()
        if not name:
            raise EEGChunkError(f"channel_names[{index}] 不能为空")
        names.append(name)

    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise EEGChunkError("channel_names 存在重复通道名：" + "、".join(duplicates))
    return tuple(names)


def _require_channel_types(values: object, count: int) -> tuple[str, ...]:
    """通道物理量类型；缺省为全 ``"eeg"``，数量必须与通道表一致。"""
    if values is None:
        return ("eeg",) * count
    if isinstance(values, (str, bytes)):
        raise EEGChunkError("channel_types 必须是通道类型序列，不能是单个字符串")
    try:
        items = list(values)  # type: ignore[arg-type]
    except TypeError:
        raise EEGChunkError("channel_types 必须是字符串序列") from None
    if len(items) != count:
        raise EEGChunkError(f"channel_types 数量应为 {count}（与通道表一致），收到 {len(items)}")

    types: list[str] = []
    for index, item in enumerate(items):
        if not isinstance(item, str):
            raise EEGChunkError(f"channel_types[{index}] 必须是字符串")
        kind = item.strip()
        if not kind:
            raise EEGChunkError(f"channel_types[{index}] 不能为空")
        types.append(kind)
    return tuple(types)


def _require_samples(values: object, channel_names: tuple[str, ...]) -> np.ndarray:
    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise EEGChunkError("samples 必须是规则的二维数值数组") from exc

    if array.ndim != 2:
        raise EEGChunkError(f"samples 必须是二维数组（通道数 × 样本数），收到 {array.ndim} 维")
    if array.shape[0] != len(channel_names):
        raise EEGChunkError(
            f"samples 第 0 维应为通道数 {len(channel_names)}，收到 {array.shape[0]}"
        )
    if array.shape[1] < 1:
        raise EEGChunkError("samples 至少要有 1 个样本")
    if not np.issubdtype(array.dtype, np.floating):
        raise EEGChunkError(
            f"samples 必须是浮点数组（内部单位为 {EEG_UNIT}），收到 dtype={array.dtype}"
        )

    stored = np.array(array, dtype=np.float64, order="C", copy=True)
    stored.flags.writeable = False
    return stored


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class EEGChunk:
    """一块已释放的脑电样本，契约版本 :data:`CONTRACT_VERSION`。

    六组信息，缺一不可：

    **身份标识**
        ``source_id`` / ``recording_id`` 数据来源与原始记录，``seq`` 记录内块编号；
        三者合成 ``chunk_id``（见 :attr:`chunk_id`），用于排重、排序与关联决策。
    **时间定位**
        ``start_sample`` 首样本零基编号，加上 ``n_samples`` 与 ``sample_rate_hz`` 得到
        源时间区间 ``[start_t_sec, source_t_sec)``；``released_monotonic_ms`` 只记录
        实际释放时刻，用于算延迟，不参与源时间。
    **形状维度**
        ``samples`` 为 ``(通道数, 样本数)`` 二维数组，对应 :attr:`shape` /
        :attr:`n_channels` / :attr:`n_samples` / :attr:`end_sample`。
    **物理属性**
        :attr:`unit` 恒为伏；数组为只读 ``float64``，非有限值原样放行。
    **通道信息**
        ``channel_names`` 是行顺序（契约不排序、不改名），``channel_types`` 是同一顺序上
        的物理量类型（首版全为 ``"eeg"``）。
    **事件信息**
        本块只带流级事件 ``discontinuity``；原始标注是另一条通道，见
        :class:`~eeg_core.contracts.events.ReferenceMarker`——标签不进这里（PRD §5.3）。

    字段以 PRD §7.1 为基线，另加 ``channel_types``（通道物理量）与 ``chunk_id``（派生）。
    构造即校验，失败抛 :class:`EEGChunkError`。
    """

    source_id: str
    recording_id: str
    seq: int
    start_sample: int
    sample_rate_hz: float
    channel_names: tuple[str, ...]
    samples: np.ndarray
    released_monotonic_ms: float
    discontinuity: bool = False
    channel_types: tuple[str, ...] | None = None
    schema_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        version = _require_text(self.schema_version, "schema_version")
        if version != CONTRACT_VERSION:
            raise EEGChunkError(
                f"不支持的 schema_version {version!r}，当前契约版本为 {CONTRACT_VERSION}"
            )
        if not isinstance(self.discontinuity, bool):
            raise EEGChunkError("discontinuity 必须是 bool")

        channel_names = _require_channel_names(self.channel_names)

        object.__setattr__(self, "schema_version", version)
        object.__setattr__(self, "source_id", _require_text(self.source_id, "source_id"))
        object.__setattr__(self, "recording_id", _require_text(self.recording_id, "recording_id"))
        object.__setattr__(self, "seq", _require_int(self.seq, "seq"))
        object.__setattr__(self, "start_sample", _require_int(self.start_sample, "start_sample"))
        object.__setattr__(
            self,
            "sample_rate_hz",
            _require_real(self.sample_rate_hz, "sample_rate_hz", strictly_positive=True),
        )
        object.__setattr__(
            self,
            "released_monotonic_ms",
            _require_real(self.released_monotonic_ms, "released_monotonic_ms", minimum=0.0),
        )
        object.__setattr__(self, "channel_names", channel_names)
        object.__setattr__(self, "samples", _require_samples(self.samples, channel_names))
        object.__setattr__(
            self,
            "channel_types",
            _require_channel_types(self.channel_types, len(channel_names)),
        )

    @property
    def shape(self) -> tuple[int, int]:
        """``(通道数, 样本数)``。"""
        return self.samples.shape

    @property
    def unit(self) -> str:
        """采样值的物理单位，恒为伏。"""
        return EEG_UNIT

    @property
    def chunk_id(self) -> str:
        """记录内可读、可排序的块标识，如 ``S001R12/000042``。"""
        return f"{self.recording_id}/{self.seq:06d}"

    @property
    def n_channels(self) -> int:
        """通道数，即 ``samples`` 的行数。"""
        return self.samples.shape[0]

    @property
    def n_samples(self) -> int:
        """块内样本数，即 ``samples`` 的列数。"""
        return self.samples.shape[1]

    @property
    def end_sample(self) -> int:
        """本块之后第一个样本的零基编号，即左闭右开区间的右端。"""
        return self.start_sample + self.n_samples

    @property
    def start_t_sec(self) -> float:
        """本块首样本的源时间，用于与参考轨对齐。"""
        return self.start_sample / self.sample_rate_hz

    @property
    def source_t_sec(self) -> float:
        """本块末尾的源时间边界 ``end_sample / sample_rate_hz``，与决策窗右端同一口径。"""
        return self.end_sample / self.sample_rate_hz

    @property
    def duration_sec(self) -> float:
        """本块覆盖的源时间长度。"""
        return self.n_samples / self.sample_rate_hz

    def has_non_finite(self) -> bool:
        """是否含 NaN / Inf；质量层据此判 ``invalid``，契约本身不清洗。"""
        return not bool(np.isfinite(self.samples).all())

    def channel_index(self, name: str) -> int:
        """按通道名取行号；不在通道表中时抛 ``KeyError``，由调用方决定是否阻止运行。"""
        try:
            return self.channel_names.index(name)
        except ValueError:
            raise KeyError(f"通道 {name!r} 不在记录 {self.recording_id} 的通道表中") from None

    def metadata(self) -> dict[str, object]:
        """不含样本的元数据，可直接写入 manifest 与 runtime 日志。"""
        return {
            "schema_version": self.schema_version,
            "source_id": self.source_id,
            "recording_id": self.recording_id,
            "seq": self.seq,
            "start_sample": self.start_sample,
            "sample_rate_hz": self.sample_rate_hz,
            "channel_names": list(self.channel_names),
            "channel_types": list(self.channel_types or ()),
            "released_monotonic_ms": self.released_monotonic_ms,
            "discontinuity": self.discontinuity,
        }

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EEGChunk):
            return NotImplemented
        return (
            self.schema_version == other.schema_version
            and self.source_id == other.source_id
            and self.recording_id == other.recording_id
            and self.seq == other.seq
            and self.start_sample == other.start_sample
            and self.sample_rate_hz == other.sample_rate_hz
            and self.channel_names == other.channel_names
            and self.channel_types == other.channel_types
            and self.discontinuity == other.discontinuity
            and self.released_monotonic_ms == other.released_monotonic_ms
            and bool(np.array_equal(self.samples, other.samples, equal_nan=True))
        )

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        return (
            f"EEGChunk(recording_id={self.recording_id!r}, seq={self.seq}, "
            f"start_sample={self.start_sample}, channels={self.n_channels}, "
            f"samples={self.n_samples}, rate_hz={self.sample_rate_hz:g}, "
            f"discontinuity={self.discontinuity})"
        )


@dataclass(frozen=True, slots=True)
class WindowRange:
    """解码观察窗 ``[start_sample, end_sample)``，左闭右开。

    窗口是「模型输入」单位，与供数块（``EEGChunk``）无关：块长怎么变，窗口都落在同一批
    样本上（T05）。三条硬规则由调用方按本类型检查：

    * **网格固定**：起点锚定记录的第 0 个样本，``start_sample = index × hop_samples``，
      用 :meth:`from_grid_index` 生成，用 :meth:`is_aligned` 复核。
    * **必须完整**：``end_sample`` 不超过已收到的有效样本（:meth:`is_complete`），
      不用未来样本补齐。
    * **不许跨断点**：整窗必须落在同一段连续有效数据里；间断或质量失败后宁可不产窗，
      也不拼一个「看起来连续」的窗。

    时间与 ``EEGChunk`` 同口径：只由样本编号和采样率推导，不碰墙钟。
    """

    sample_rate_hz: float
    start_sample: int
    end_sample: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "sample_rate_hz",
            _require_real(self.sample_rate_hz, "sample_rate_hz", strictly_positive=True),
        )
        start = _require_int(self.start_sample, "start_sample")
        end = _require_int(self.end_sample, "end_sample", minimum=1)
        if end <= start:
            raise EEGChunkError(
                f"end_sample 必须大于 start_sample，收到 [{start}, {end})"
            )
        object.__setattr__(self, "start_sample", start)
        object.__setattr__(self, "end_sample", end)

    @classmethod
    def from_grid_index(
        cls,
        index: int,
        *,
        sample_rate_hz: float,
        window_samples: int,
        hop_samples: int,
        anchor_sample: int = 0,
    ) -> WindowRange:
        """按网格生成第 ``index`` 个窗口；网格锚点默认是记录的样本 0。"""
        position = _require_int(index, "index")
        window = _require_int(window_samples, "window_samples", minimum=1)
        hop = _require_int(hop_samples, "hop_samples", minimum=1)
        anchor = _require_int(anchor_sample, "anchor_sample")
        start = anchor + position * hop
        return cls(sample_rate_hz=sample_rate_hz, start_sample=start, end_sample=start + window)

    @property
    def n_samples(self) -> int:
        """窗口内的样本数。"""
        return self.end_sample - self.start_sample

    @property
    def start_t_sec(self) -> float:
        """窗口首样本的源时间。"""
        return self.start_sample / self.sample_rate_hz

    @property
    def source_t_sec(self) -> float:
        """窗口末端的源时间边界，与 ``DecisionFrame.source_t_sec`` 同一口径。"""
        return self.end_sample / self.sample_rate_hz

    @property
    def duration_sec(self) -> float:
        """窗口覆盖的源时间长度。"""
        return self.n_samples / self.sample_rate_hz

    def contains_sample(self, sample: int) -> bool:
        """左闭右开：``start <= sample < end``。"""
        return self.start_sample <= sample < self.end_sample

    def overlaps(self, other: WindowRange) -> bool:
        """两个窗口是否有交集；相邻（首尾相接）不算重叠。"""
        return self.start_sample < other.end_sample and other.start_sample < self.end_sample

    def is_complete(self, available_end_sample: int) -> bool:
        """``available_end_sample`` 是已收到数据之后第一个样本的编号（左闭右开）。"""
        available = _require_int(available_end_sample, "available_end_sample")
        return self.end_sample <= available

    def is_aligned(self, hop_samples: int, anchor_sample: int = 0) -> bool:
        """起点是否落在网格上；锚点默认是记录的样本 0。"""
        hop = _require_int(hop_samples, "hop_samples", minimum=1)
        anchor = _require_int(anchor_sample, "anchor_sample")
        return (self.start_sample - anchor) % hop == 0


@dataclass(frozen=True, slots=True)
class QualityReport:
    """一次质量检查的结论：能不能用、为什么不能用、哪些通道有问题。

    ``reasons`` 只能取 :data:`QUALITY_REASONS` 里的原因码，保证界面、CSV 与测试说的是
    同一件事。``valid`` 与 ``reasons`` 互斥：合格就不许带原因，不合格就必须给原因。

    ``window`` 用于窗口级检查；块级快速检查（非有限值、缺通道）没有窗口，留 ``None``。
    """

    valid: bool
    reasons: tuple[str, ...] = ()
    bad_channels: tuple[str, ...] = ()
    window: WindowRange | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.valid, bool):
            raise EEGChunkError("valid 必须是 bool")
        if self.reasons is None or isinstance(self.reasons, (str, bytes)):
            raise EEGChunkError("reasons 必须是原因码序列（不能为 None 或单个字符串）")
        if self.bad_channels is None or isinstance(self.bad_channels, (str, bytes)):
            raise EEGChunkError("bad_channels 必须是通道名序列（不能为 None 或单个字符串）")
        if self.window is not None and not isinstance(self.window, WindowRange):
            raise EEGChunkError("window 必须是 WindowRange 或 None")

        reasons: list[str] = []
        for index, item in enumerate(tuple(self.reasons)):
            if not isinstance(item, str) or item not in QUALITY_REASONS:
                raise EEGChunkError(
                    f"reasons[{index}] 必须是 {QUALITY_REASONS} 之一，收到 {item!r}"
                )
            if item in reasons:
                raise EEGChunkError(f"reasons 重复：{item}")
            reasons.append(item)

        channels: list[str] = []
        for index, item in enumerate(tuple(self.bad_channels)):
            if not isinstance(item, str) or not item.strip():
                raise EEGChunkError(f"bad_channels[{index}] 必须是非空字符串")
            name = item.strip()
            if name in channels:
                raise EEGChunkError(f"bad_channels 重复：{name}")
            channels.append(name)

        if self.valid and (reasons or channels):
            raise EEGChunkError("valid=True 时不能带 reasons 或 bad_channels")
        if not self.valid and not reasons:
            raise EEGChunkError("valid=False 时必须给出至少一个原因码")

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
    """逐窗决策的六种状态（PRD §6.3）。

    运行状态（Ready/Running/…）是另一回事，两者分开保存，不要混用。
    """

    WARMING_UP = "warming_up"
    """历史不足（刚启动或刚重置），不下结论。"""

    IDLE = "idle"
    """空闲。注意：它是模型学到的"休息"类别，不等于任意场景下的空闲。"""

    LEFT_HAND = "left_hand"
    RIGHT_HAND = "right_hand"
    """左右手运动想象，只有这两个状态会驱动动画。"""

    UNCERTAIN = "uncertain"
    """数据合格但门限/时间一致性没过，拒判。**低分不能直接转成 idle。**"""

    INVALID = "invalid"
    """数据质量不合格或流中断，当前结果不可用。"""

    @property
    def is_action(self) -> bool:
        """是否为会驱动动画的动作状态（左手或右手）。"""
        return self in (DecisionState.LEFT_HAND, DecisionState.RIGHT_HAND)


@dataclass(frozen=True, slots=True)
class Scores:
    """三状态分数。

    取值 0~1、必须是有限数；**不要求相加等于 1**——类型是未校准概率，不能当置信度解释。
    """

    idle: float
    left_hand: float
    right_hand: float

    def __post_init__(self) -> None:
        for name in ("idle", "left_hand", "right_hand"):
            value = _require_real(getattr(self, name), name, minimum=0.0)
            if value > 1.0:
                raise EEGChunkError(f"{name} 不能大于 1，收到 {value!r}")
            object.__setattr__(self, name, value)

    def as_dict(self) -> dict[str, float]:
        """按 idle、left_hand、right_hand 的顺序返回字典。"""
        return {"idle": self.idle, "left_hand": self.left_hand, "right_hand": self.right_hand}

    def top(self) -> tuple[str, float]:
        """最高分类别与分值；并列时按字典顺序取第一个。"""
        return max(self.as_dict().items(), key=lambda item: item[1])


@dataclass(frozen=True, slots=True)
class DecisionFrame:
    """一个更新点的决策结果（B → C/D），字段与 PRD §7.2 的 JSON 示例一一对应。

    一次决策 = 一个证据窗 + 一个质量结论 + 一个状态 + （可选）分数。

    校验规则（构造时强制）：

    * ``schema_version`` 必须是当前契约版本。
    * ``warming_up``：没有完整窗口、没有分数、没有 intent。
    * ``invalid``：必须有不合格的质量报告；窗口可以有（窗内质量失败）也可以没有（流中断）。
    * ``idle`` / ``uncertain`` / ``left_hand`` / ``right_hand``：必须有窗口、必须有分数、质量必须合格。
    * ``intent`` 只能出现且等于 ``left_hand`` 或 ``right_hand``；其他状态一律为 ``None``。
    * 有窗口时，``sample_rate_hz`` 与 ``source_t_sec`` 必须和窗口一致（末端边界时间）。
    * ``generated_at_utc`` 是 UTC 的 ISO 时间；``emitted_monotonic_ms`` 是会话单调时钟，
      两者不能混用。
    * 全部数值必须是有限值，写出的 JSON 不允许出现 NaN / Infinity。
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
            raise EEGChunkError(
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
            raise EEGChunkError(
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
            raise EEGChunkError(
                f"decision_state 必须是 DecisionState，收到 {type(self.decision_state).__name__}"
            )
        if self.scores is not None and not isinstance(self.scores, Scores):
            raise EEGChunkError("scores 必须是 Scores 或 None")
        if not isinstance(self.quality, QualityReport):
            raise EEGChunkError("quality 必须是 QualityReport")
        if self.intent is not None and not isinstance(self.intent, DecisionState):
            raise EEGChunkError("intent 必须是 DecisionState 或 None")
        if self.window is not None and not isinstance(self.window, WindowRange):
            raise EEGChunkError("window 必须是 WindowRange 或 None")

        state = self.decision_state
        self._check_state_rules(state)
        self._check_window(state)

    def _check_state_rules(self, state: DecisionState) -> None:
        if self.intent is not None:
            if not self.intent.is_action:
                raise EEGChunkError(
                    f"intent 只能是 left_hand 或 right_hand，收到 {self.intent.value!r}"
                )
            if self.intent is not state:
                raise EEGChunkError(
                    f"intent 与 decision_state 不一致：{self.intent.value} vs {state.value}"
                )
        elif state.is_action:
            raise EEGChunkError(f"{state.value} 状态必须给出 intent")

        if state.is_action or state in (DecisionState.IDLE, DecisionState.UNCERTAIN):
            if self.scores is None:
                raise EEGChunkError(f"{state.value} 状态必须有模型分数")
        elif self.scores is not None:
            raise EEGChunkError(f"{state.value} 状态不应带模型分数")

        if state is DecisionState.INVALID:
            if self.quality.valid:
                raise EEGChunkError("invalid 状态必须带不合格的质量报告")
        elif not self.quality.valid:
            raise EEGChunkError("只有 invalid 状态可以带不合格的质量报告")

    def _check_window(self, state: DecisionState) -> None:
        if state is DecisionState.WARMING_UP:
            if self.window is not None:
                raise EEGChunkError("warming_up 状态不应带完整窗口")
            return
        if self.window is None:
            if state is not DecisionState.INVALID:
                raise EEGChunkError(f"{state.value} 状态必须带证据窗")
            return
        if abs(self.window.sample_rate_hz - self.sample_rate_hz) > 1e-9:
            raise EEGChunkError("window 的采样率与 sample_rate_hz 不一致")
        if abs(self.window.source_t_sec - self.source_t_sec) > 1e-9:
            raise EEGChunkError(
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
