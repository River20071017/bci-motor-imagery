"""契约 1.0 的输入侧数据：一块脑电、一个观察窗、一条记录的档案。

本模块只放「进来的数据」和它们的校验，不放协议、不放输出结构、不放配置。
三个盒子都是 ``frozen`` dataclass：构造即校验，交出去就改不了。

全包统一的两条换算规则：

* **单位**：``EEGChunk.samples`` 一律为伏（``EEG_UNIT``）。适配器负责把 EDF 的微伏换算成伏，
  界面再乘 ``1e6`` 显示；中间任何一层都不再换算。
* **时间**：只由样本编号和采样率推导（``样本号 / sample_rate_hz``），不碰墙钟。

块里没有标签：原始标注在 :mod:`eeg_core.contracts.events`，单独一条线（PRD §5.3）。
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Sequence

from dataclasses import dataclass

import numpy as np

CONTRACT_VERSION = "1.0"
"""当前契约版本；字段或语义变化必须同时升级版本并更新 docs/01 的两张表。"""

EEG_UNIT = "V"
"""``EEGChunk.samples`` 的内部单位，固定为伏。"""

__all__ = [
    "CONTRACT_VERSION",
    "EEG_UNIT",
    "ContractError",
    "IncompatibleSourceError",
    "EEGChunk",
    "SourceInfo",
    "WindowRange",
]


class ContractError(ValueError):
    """任何契约盒子不满足字段约束时抛出的唯一异常。

    调用方只认这一种：``except ContractError`` 就够，不用记住十几个子类。
    子类只用来区分场景（如 :class:`IncompatibleSourceError`），不改变处理方式：
    都属于「数据不可用」，应阻止运行并说明原因，不猜测修正（T02）。
    """


def _require_text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise ContractError(f"{field_name} 必须是字符串，收到 {type(value).__name__}")
    text = value.strip()
    if not text:
        raise ContractError(f"{field_name} 不能为空")
    return text


def _require_int(value: object, field_name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ContractError(f"{field_name} 必须是整数，收到 {type(value).__name__}")
    number = int(value)
    if number < minimum:
        raise ContractError(f"{field_name} 不能小于 {minimum}，收到 {number}")
    return number


def _require_real(
    value: object,
    field_name: str,
    *,
    minimum: float | None = None,
    strictly_positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ContractError(f"{field_name} 必须是实数，收到 {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise ContractError(f"{field_name} 必须是有限值，收到 {number!r}")
    if strictly_positive and number <= 0.0:
        raise ContractError(f"{field_name} 必须大于 0，收到 {number!r}")
    if minimum is not None and number < minimum:
        raise ContractError(f"{field_name} 不能小于 {minimum}，收到 {number!r}")
    return number


def _require_channel_names(values: object) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ContractError("channel_names 必须是通道名序列，不能是单个字符串")
    try:
        items = list(values)  # type: ignore[arg-type]
    except TypeError:
        raise ContractError("channel_names 必须是字符串序列") from None
    if not items:
        raise ContractError("channel_names 不能为空")

    names: list[str] = []
    for index, item in enumerate(items):
        if not isinstance(item, str):
            raise ContractError(f"channel_names[{index}] 必须是字符串")
        name = item.strip()
        if not name:
            raise ContractError(f"channel_names[{index}] 不能为空")
        names.append(name)

    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ContractError("channel_names 存在重复通道名：" + "、".join(duplicates))
    return tuple(names)


def _require_channel_types(values: object, count: int) -> tuple[str, ...]:
    """通道物理量类型；缺省为全 ``"eeg"``，数量必须与通道表一致。"""
    if values is None:
        return ("eeg",) * count
    if isinstance(values, (str, bytes)):
        raise ContractError("channel_types 必须是通道类型序列，不能是单个字符串")
    try:
        items = list(values)  # type: ignore[arg-type]
    except TypeError:
        raise ContractError("channel_types 必须是字符串序列") from None
    if len(items) != count:
        raise ContractError(f"channel_types 数量应为 {count}（与通道表一致），收到 {len(items)}")

    types: list[str] = []
    for index, item in enumerate(items):
        if not isinstance(item, str):
            raise ContractError(f"channel_types[{index}] 必须是字符串")
        kind = item.strip()
        if not kind:
            raise ContractError(f"channel_types[{index}] 不能为空")
        types.append(kind)
    return tuple(types)


def _require_samples(values: object, channel_names: tuple[str, ...]) -> np.ndarray:
    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ContractError("samples 必须是规则的二维数值数组") from exc

    if array.ndim != 2:
        raise ContractError(f"samples 必须是二维数组（通道数 × 样本数），收到 {array.ndim} 维")
    if array.shape[0] != len(channel_names):
        raise ContractError(
            f"samples 第 0 维应为通道数 {len(channel_names)}，收到 {array.shape[0]}"
        )
    if array.shape[1] < 1:
        raise ContractError("samples 至少要有 1 个样本")
    if not np.issubdtype(array.dtype, np.floating):
        raise ContractError(
            f"samples 必须是浮点数组（内部单位为 {EEG_UNIT}），收到 dtype={array.dtype}"
        )

    stored = np.array(array, dtype=np.float64, order="C", copy=True)
    stored.flags.writeable = False
    return stored


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class EEGChunk:
    """一块已释放的脑电样本，契约版本 :data:`CONTRACT_VERSION`。

    六组信息，缺一不可：

    **身份标识** ``source_id`` / ``recording_id`` / ``seq``，三者合成 :attr:`chunk_id`。
    **时间定位** ``start_sample`` + ``n_samples`` + ``sample_rate_hz`` 推出源时间区间；
    ``released_monotonic_ms`` 只记录释放时刻，用于算延迟，不参与源时间。
    **形状维度** ``samples`` 为 ``(通道数, 样本数)``，对应 :attr:`shape` / :attr:`n_channels` /
    :attr:`n_samples` / :attr:`end_sample`。
    **物理属性** :attr:`unit` 恒为伏；数组只读 ``float64``，非有限值原样放行。
    **通道信息** ``channel_names`` 是行顺序（不排序、不改名），``channel_types`` 是同一顺序的
    物理量类型（首版全为 ``"eeg"``）。
    **事件信息** 只有流级事件 ``discontinuity``；原始标注不进这里。

    构造即校验，失败抛 :class:`ContractError`。
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
            raise ContractError(
                f"不支持的 schema_version {version!r}，当前契约版本为 {CONTRACT_VERSION}"
            )
        if not isinstance(self.discontinuity, bool):
            raise ContractError("discontinuity 必须是 bool")

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
        """本块首样本的源时间。"""
        return self.start_sample / self.sample_rate_hz

    @property
    def source_t_sec(self) -> float:
        """本块末尾的源时间边界 ``end_sample / sample_rate_hz``，与决策窗右端同口径。"""
        return self.end_sample / self.sample_rate_hz

    @property
    def duration_sec(self) -> float:
        """本块覆盖的源时间长度。"""
        return self.n_samples / self.sample_rate_hz

    def has_non_finite(self) -> bool:
        """是否含 NaN / Inf；质量层据此判 ``invalid``，契约本身不清洗。"""
        return not bool(np.isfinite(self.samples).all())

    def channel_index(self, name: str) -> int:
        """按通道名取行号；没有这个通道时抛 ``KeyError``。"""
        try:
            return self.channel_names.index(name)
        except ValueError:
            raise KeyError(f"通道 {name!r} 不在记录 {self.recording_id} 的通道表中") from None

    def metadata(self) -> dict[str, object]:
        """不含样本的元数据，可直接写 manifest 与运行日志。"""
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

    窗口是「模型输入」单位，与供数块无关：块长怎么变，窗口都落在同一批样本上（T05）。
    四条规则由调用方按本类型检查：

    * **网格固定**：起点锚定记录的样本 0，``start = index × hop``，用 :meth:`from_grid_index`
      生成、:meth:`is_aligned` 复核。
    * **必须完整**：``end_sample`` 不超过已收到的有效样本（:meth:`is_complete`）。
    * **不许跨断点**：整窗落在同一段连续有效数据里，间断后宁可不产窗。
    * **先预热**：该段连续有效满 3 秒才出第一个窗（由产窗逻辑保证）。

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
            raise ContractError(f"end_sample 必须大于 start_sample，收到 [{start}, {end})")
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
class SourceInfo:
    """一条原始记录的档案卡：记录是什么，不是能不能解码。

    ``reference`` 为 ``None`` 表示记录没有写明的参考方式——按规范标为「未提供」，不猜。
    兼容性判断在 :meth:`check_compatible` 里按调用方要求逐项核对，不在数据层写死模型参数。
    """

    source_id: str
    recording_id: str
    sample_rate_hz: float
    channel_names: tuple[str, ...]
    reference: str | None = None
    subject_id: str | None = None
    run: str | None = None
    n_samples: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _require_text(self.source_id, "source_id"))
        object.__setattr__(self, "recording_id", _require_text(self.recording_id, "recording_id"))
        object.__setattr__(
            self,
            "sample_rate_hz",
            _require_real(self.sample_rate_hz, "sample_rate_hz", strictly_positive=True),
        )
        object.__setattr__(self, "channel_names", _require_channel_names(self.channel_names))
        object.__setattr__(self, "reference", _optional_text(self.reference, "reference"))
        object.__setattr__(self, "subject_id", _optional_text(self.subject_id, "subject_id"))
        object.__setattr__(self, "run", _optional_text(self.run, "run"))
        if self.n_samples is not None:
            object.__setattr__(self, "n_samples", _require_int(self.n_samples, "n_samples"))

    @property
    def reference_label(self) -> str:
        """界面与 manifest 用的参考方式文案；未提供时明确写「未提供」。"""
        return self.reference if self.reference is not None else "未提供"

    @property
    def duration_sec(self) -> float | None:
        """记录总时长；``n_samples`` 未知时为 ``None``，不估算。"""
        if self.n_samples is None:
            return None
        return self.n_samples / self.sample_rate_hz

    def check_compatible(
        self,
        *,
        required_channels: Sequence[str] = (),
        sample_rate_hz: float | None = None,
        required_reference: str | None = None,
        sample_rate_tolerance_hz: float = 1e-3,
    ) -> None:
        """按调用方要求核对本记录，不通过时一次性抛出全部原因。

        只判断「能不能用」，不修改任何东西：不补通道、不重采样、不猜测参考方式。
        ``sample_rate_hz`` 与 ``required_reference`` 传 ``None`` 表示该项不作要求。
        """
        if isinstance(required_channels, str):
            raise TypeError("required_channels 必须是通道名序列，不能是单个字符串")

        problems: list[str] = []

        available = set(self.channel_names)
        missing = [name for name in required_channels if name not in available]
        if missing:
            problems.append("缺少模型必需通道：" + "、".join(missing))

        if sample_rate_hz is not None:
            expected = _require_real(sample_rate_hz, "sample_rate_hz", strictly_positive=True)
            if abs(self.sample_rate_hz - expected) > sample_rate_tolerance_hz:
                problems.append(
                    f"采样率不符：记录为 {self.sample_rate_hz:g} Hz，要求 {expected:g} Hz"
                )

        if required_reference is not None:
            if self.reference is None:
                problems.append(
                    f"记录未提供参考方式，无法确认与要求的 {required_reference!r} 一致"
                )
            elif self.reference != required_reference:
                problems.append(
                    f"参考方式不符：记录为 {self.reference!r}，要求 {required_reference!r}"
                )

        if problems:
            raise IncompatibleSourceError(self.recording_id, problems)


class IncompatibleSourceError(ContractError):
    """记录与模型或配置不兼容，一次性携带全部原因（T02：阻止运行并写清原因）。"""

    def __init__(self, recording_id: str, reasons: Sequence[str]) -> None:
        self.recording_id = recording_id
        self.reasons = tuple(reasons)
        if not self.reasons:
            raise ValueError("IncompatibleSourceError 至少需要一个原因")
        super().__init__(f"记录 {recording_id} 不可用：" + "；".join(self.reasons))


def _optional_text(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_text(value, field_name)
