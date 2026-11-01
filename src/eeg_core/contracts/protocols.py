"""数据源契约：从一条原始记录到 :class:`~eeg_core.contracts.frames.EEGChunk` 的接缝。

``sources/`` 下的适配器（首版是 ``data/`` 里 PhysioNet EDF+ 的回放）实现
:class:`DataSource`；``runtime`` 与 ``processing`` 只依赖本模块的协议、``SourceInfo`` 和
``EEGChunk``，不 import mne，也不认识文件路径。这样换数据源不改算法与界面（F01 / T02）。

适配器必须满足的不变量：

1. 一个实例只服务一条记录。换记录就换实例；``seq`` 从 0 重新开始，且首块
   ``discontinuity=True`` 标出新记录边界。
2. 块按 ``seq`` 递增、``start_sample`` 首尾相接；块内样本数可以变化（验收 T05：同一段
   样本按不同块长供给，窗口边界与结果一致）。已知间断（丢包、暂停后不补齐、重连）必须
   置 ``discontinuity=True``，不得伪装成连续样本。
3. 只暴露已释放的过去样本；调用方拿不到未来样本，也不能用整文件预先算好的特征。
4. 不含类别、试次编号或参考标注时间表（PRD §5.3 标签隔离）。
5. 记录结束返回 ``None``，且之后继续返回 ``None``；``close()`` 之后调用 ``read_chunk()``
   抛 :class:`SourceClosedError`。
6. 原始标注不随块下发：参考轨需要的标注走 :class:`MarkerSource`，解码路径拿不到。

切分是否合法用 :class:`ChunkStreamGuard` 逐块检查（适配器单测里挂一行即可）。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, Self, runtime_checkable

from .events import ReferenceMarker
from .frames import (
    EEGChunk,
    _require_channel_names,
    _require_int,
    _require_real,
    _require_text,
)

__all__ = [
    "ChunkSequenceError",
    "ChunkStreamGuard",
    "DataSource",
    "IncompatibleSourceError",
    "MarkerSource",
    "ResetReason",
    "SourceClosedError",
    "SourceError",
    "SourceInfo",
    "StreamConsumer",
]


class SourceError(RuntimeError):
    """数据源层面的错误基类：打开失败、读取失败、关闭后继续读取等。"""


class SourceClosedError(SourceError):
    """数据源已关闭，不能再读取。"""


class IncompatibleSourceError(SourceError):
    """记录与模型或配置不兼容。

    一次性携带全部原因，供界面直接说明（T02：阻止运行并写清原因，不猜测修正）。
    """

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


@dataclass(frozen=True, slots=True)
class SourceInfo:
    """一条原始记录的元数据，在第一次 ``read_chunk()`` 之前即可读取。

    只描述「记录是什么」，不描述「能不能解码」：兼容性判断在 :meth:`check_compatible`
    里按模型包要求逐项核对，避免把某个个人模型的参数写死进数据层。

    ``reference`` 为 ``None`` 表示记录没有写明的参考方式——按规范标为「未提供」，
    不猜测、不自动补一个。
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
        """按模型包要求核对本记录，不通过时一次性抛出全部原因。

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


@runtime_checkable
class DataSource(Protocol):
    """从一条记录连续供数的适配器接口（首版实现：EDF 回放，见 ``sources/``）。

    ``__enter__`` / ``__exit__`` 让文件句柄与回放线程有确定的释放点：

    .. code-block:: python

        with EdfReplaySource(path, info=info) as source:
            while (chunk := source.read_chunk()) is not None:
                handle(chunk)
    """

    @property
    def info(self) -> SourceInfo:
        """记录元数据；与块内容无关，可反复读取。"""
        ...

    def read_chunk(self) -> EEGChunk | None:
        """返回下一块；记录结束返回 ``None``，且之后一直返回 ``None``。"""
        ...

    def close(self) -> None:
        """释放资源，可重复调用；关闭后 ``read_chunk()`` 抛 :class:`SourceClosedError`。"""
        ...

    def __enter__(self) -> Self:
        """返回自身；进入后即可读取。"""
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object,
    ) -> None:
        """退出时调用 ``close()``，不吞掉异常。"""
        ...


@runtime_checkable
class MarkerSource(Protocol):
    """原始标注的提供方，只给参考轨与评估用。

    实现通常与 :class:`DataSource` 是同一个适配器对象，但**不得**注入解码路径：
    解码器只接收 ``DataSource`` / ``EEGChunk``，拿不到 ``markers``（PRD §5.3、T03）。
    """

    @property
    def markers(self) -> tuple[ReferenceMarker, ...]:
        """按 ``onset_sample`` 升序排列的原始标注；不解释、不映射类别。"""
        ...


class ResetReason(StrEnum):
    """请求重置的原因；不同原因要求清理的状态不同（见 :class:`StreamConsumer`）。

    用字符串枚举是为了能直接写进 ``runtime.jsonl``，不必再维护一套映射。
    """

    SESSION_START = "session_start"
    """新会话：一切都是新的。"""

    DISCONTINUITY = "discontinuity"
    """块上标了 ``discontinuity=True``：新记录边界或已知间断，样本不再连续。"""

    QUALITY_INVALID = "quality_invalid"
    """质量检查不合格：受影响的历史不能继续参与后续窗口。"""

    PAUSE_RESUME = "pause_resume"
    """用户暂停后继续：数据没有断，但暂停前的候选确认必须作废，不能重复发布旧窗。"""


@runtime_checkable
class StreamConsumer(Protocol):
    """有状态的流式消费者：滤波器、窗器、特征器、解码器都实现它。

    运行时会话控制器在把手里的下一块交给消费者**之前**调用 :meth:`reset`，把「什么时候
    必须忘掉历史」从口头约定变成必须实现的方法。忘了重置的后果是隐蔽的：滤波器会把断点
    前后的样本算成一段连续信号，结果看起来完全正常，但错得没边（T04 / T07）。

    各原因要求的效果：

    ``SESSION_START`` / ``DISCONTINUITY`` / ``QUALITY_INVALID``
        丢弃全部与样本有关的状态——滤波状态（``zi``）、未完成的窗口、预热计数、候选确认
        计数、当前动作的连续判定；效果等同于「从新数据的第一块重新开始」。在下一次预热
        满足之前，输出必须是 ``warming_up`` 或 ``invalid``，不得沿用旧状态。

    ``PAUSE_RESUME``
        保留滤波状态与已处理历史（数据没有断），只作废候选确认计数，避免把暂停前已经
        发布过的窗重复发布。

    ``reset`` 必须可重复调用（幂等），且不得抛异常干扰会话控制。
    """

    def reset(self, reason: ResetReason) -> None:
        """按原因清理内部状态。"""
        ...


class ChunkSequenceError(SourceError):
    """块序列违反契约：跳号、样本不连续、中途换元数据、首块没标新记录边界。"""


class ChunkStreamGuard:
    """逐块检查适配器的切分是否合法，给适配器单测与集成测试用。

    .. code-block:: python

        guard = ChunkStreamGuard(source.info)
        while (chunk := source.read_chunk()) is not None:
            guard.accept(chunk)

    只查结构自洽，**不查块长**——同一段样本按 8 点还是 10 点一块都必须通过（T05）。
    规则：首块 ``seq=0`` 且 ``discontinuity=True``（新记录边界）；之后 ``seq`` 严格 +1；
    未标间断的块必须 ``start_sample == 上一块.end_sample``；``recording_id``、
    ``sample_rate_hz``、``channel_names``、``channel_types`` 整条记录内不得变化。
    """

    def __init__(self, info: SourceInfo) -> None:
        self._info = info
        self._expected_seq = 0
        self._expected_start = 0
        self._seen_any = False
        self._channel_types: tuple[str, ...] | None = None

    @property
    def expected_seq(self) -> int:
        """下一块应有的 ``seq``。"""
        return self._expected_seq

    @property
    def expected_start_sample(self) -> int:
        """下一块应有的 ``start_sample``；标了间断的块可以跳过。"""
        return self._expected_start

    def accept(self, chunk: EEGChunk) -> EEGChunk:
        """校验并记账，返回原块；不合法时抛 :class:`ChunkSequenceError`。"""
        problems: list[str] = []

        if chunk.recording_id != self._info.recording_id:
            problems.append(
                f"记录标识不一致：期望 {self._info.recording_id!r}，收到 {chunk.recording_id!r}"
            )
        if chunk.sample_rate_hz != self._info.sample_rate_hz:
            problems.append(
                f"采样率中途变化：期望 {self._info.sample_rate_hz:g} Hz，"
                f"收到 {chunk.sample_rate_hz:g} Hz"
            )
        if chunk.channel_names != self._info.channel_names:
            problems.append("通道表中途变化：顺序或名称与 SourceInfo 不一致")
        if self._channel_types is None:
            self._channel_types = chunk.channel_types
        elif chunk.channel_types != self._channel_types:
            problems.append("通道类型中途变化：与首个块的类型表不一致")

        if chunk.seq != self._expected_seq:
            problems.append(f"块编号不连续：期望 seq={self._expected_seq}，收到 {chunk.seq}")

        if not self._seen_any:
            if not chunk.discontinuity:
                problems.append("记录首块必须标 discontinuity=True（新记录边界）")
        elif chunk.discontinuity:
            if chunk.start_sample < self._expected_start:
                problems.append(
                    f"间断块与上一块重叠：起点 {chunk.start_sample} 早于 {self._expected_start}"
                )
        elif chunk.start_sample != self._expected_start:
            delta = chunk.start_sample - self._expected_start
            kind = "重叠" if delta < 0 else "缺口"
            problems.append(
                f"块起点不连续：期望 start_sample={self._expected_start}，"
                f"收到 {chunk.start_sample}（{kind} {abs(delta)} 个样本）"
            )

        if problems:
            raise ChunkSequenceError(
                f"块序列不合法（{chunk.recording_id} seq={chunk.seq}）：" + "；".join(problems)
            )

        self._seen_any = True
        self._expected_seq = chunk.seq + 1
        self._expected_start = chunk.end_sample
        return chunk
