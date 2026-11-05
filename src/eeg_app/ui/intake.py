"""应用层数据接入校验（C 分工 2.x）——把"固定输入格式"落到用户可感知的弹窗之前，
先在边界做一次契约校验。

- 底层固定格式由团队契约 ``eeg_core.contracts.EEGChunk`` 定义（单位伏、只读 float64、
  ``(通道数, 样本数)`` 二维），并由 ``ChunkStreamGuard`` 逐块校验序列。
- 本模块只"调用"契约提供的校验器，不重复定义格式（硬边界①：契约只由 A 合并）。
- 返回 :class:`IntakeResult`，由 ``MainWindow`` 决定弹"确认"还是"报错"对话框（见 item 2）。

兼容说明：当前 ``main`` 契约与 B 冻结分支在 ``ChunkStreamGuard`` 的方法名
（``accept`` / ``validate``）和异常类型上不完全一致，本模块用防御式写法兼容两边，
避免 C 的 UI 随契约小版本变动而硬断裂（留后手）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from eeg_core.contracts import ChunkStreamGuard, EEGChunk, SourceInfo

# 兼容：冻结契约把 EEGChunkError 改名为 ContractError；两个都兜住。
try:
    from eeg_core.contracts import ContractError
except ImportError:  # 旧契约仍是 EEGChunkError
    from eeg_core.contracts import EEGChunkError as ContractError

from .playback import SourceProtocol


@dataclass
class IntakeResult:
    """接入校验结果：``ok=True`` 应用层应弹确认框；``ok=False`` 应弹报错框。"""

    ok: bool
    message: str
    info: "SourceInfo | None" = None


def check_source_contract(source: SourceProtocol) -> IntakeResult:
    """校验一个数据源是否满足团队固定输入格式（EEGChunk 契约）。

    这是 item 1（底层可接取固定格式）在应用边界的"可确认"入口：底层代码（A 的
    数据源 / 适配器）负责产出合法 ``EEGChunk``，本函数负责在交给 UI 前做一次把关。
    """
    try:
        info = source.info
        first = source.read_chunk()
        if first is None:
            return IntakeResult(
                False, "数据源未产生任何数据块（首块为 None，可能尚未开始或已结束）"
            )
        if not isinstance(first, EEGChunk):
            return IntakeResult(
                False,
                f"首块不是团队固定格式 EEGChunk，而是 {type(first).__name__}"
                "（底层未产出 EEGChunk）",
            )
        # 最佳努力：首块序列边界校验（不同契约版本方法名不同：validate / accept）。
        try:
            guard = ChunkStreamGuard(info)
            checker = getattr(guard, "validate", None) or getattr(guard, "accept", None)
            if checker is not None:
                checker(first)
        except Exception as seq_exc:  # noqa: BLE001
            return IntakeResult(False, f"首块序列不符合契约：{seq_exc}")
        return IntakeResult(
            True,
            f"数据格式正确：{len(info.channel_names)} 通道 / "
            f"{info.sample_rate_hz:g} Hz / 记录 {info.recording_id}",
            info,
        )
    except ContractError as exc:
        return IntakeResult(False, f"数据格式不符合团队契约（EEGChunk 构造校验失败）：{exc}")
    except Exception as exc:  # noqa: BLE001
        return IntakeResult(False, f"读取数据源时发生异常：{exc!r}")
