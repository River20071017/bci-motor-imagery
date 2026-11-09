"""契约 1.0 唯一的接口：数据校验器。

按纪律，这里**不**提前抽象数据源、解码器、存档等插座——只有真的出现「离线文件」和
「在线实时」两套实现时才加协议。当前只保留一个校验器接口，供适配器、测试和集成入口复用。

``ChunkStreamGuard`` 是它的参考实现：逐块校验并记住上一块的位置，专门抓「跳号、缺口、
重叠、中途换元数据、首块没标新记录边界」这些切分错误。它**不检查块长**——同一段样本按
8 点还是 10 点一块都必须通过（T05）。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .frames import ContractError, EEGChunk, SourceInfo

__all__ = ["ChunkSequenceError", "ChunkStreamGuard", "DataValidator"]


@runtime_checkable
class DataValidator(Protocol):
    """数据校验器：不合格就抛 :class:`ContractError`，合格返回原对象。

    只定义一个方法，是因为当前真正需要复用的就是「校验」；等有了第二套数据源或真正的
    在线实现，再讨论要不要加别的接口。
    """

    def validate(self, chunk: EEGChunk) -> EEGChunk:
        """校验一块数据并返回它；不合法时抛 :class:`ContractError`。"""
        ...


class ChunkSequenceError(ContractError):
    """块序列违反契约：跳号、样本不连续、中途换元数据、首块没标新记录边界。"""


class ChunkStreamGuard:
    """逐块检查切分是否合法，给适配器单测与集成测试用。

    .. code-block:: python

        guard = ChunkStreamGuard(info)
        while (chunk := source.read_chunk()) is not None:
            guard.validate(chunk)

    规则（与 docs/01 输入字段表一致）：

    * 首块 ``seq=0`` 且 ``discontinuity=True``（新记录边界）；
    * 之后 ``seq`` 严格 +1；
    * 未标间断的块必须 ``start_sample == 上一块.end_sample``；间断块不能往回跳；
    * ``recording_id``、``sample_rate_hz``、``channel_names``、``channel_types`` 不得中途变化。
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

    def validate(self, chunk: EEGChunk) -> EEGChunk:
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
