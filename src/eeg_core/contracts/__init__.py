"""契约 1.0：全项目的数据盒子与唯一接口。

五个模块，各管一件事，互不抢活：

=========================  ==========================================
``frames.py``              输入侧数据：``EEGChunk`` / ``WindowRange`` / ``SourceInfo``
``outputs.py``             输出侧数据：``QualityReport`` / ``Scores`` / ``DecisionFrame`` 等
``events.py``              事件：``ReferenceMarker``（参考标注）、``ResetReason``（重置原因）
``protocols.py``           唯一的接口：``DataValidator`` + 实现 ``ChunkStreamGuard``
``config.py``              全局参数唯一出处：``AppConfig`` + ``DEFAULT_CONFIG``
=========================  ==========================================

三条纪律（改代码前先读）：

1. 不提前抽象：``protocols.py`` 只有一个校验器接口，没有数据源/解码器/存档插座。
2. 锁死数据：所有盒子都是 ``frozen`` dataclass，流水线中途改不了。
3. 文档贴脸放：字段表在 ``docs/01_接口约定_1.0.md``，代码按表格写，两边一起改。

跨模块唯一需要记住的异常是 :class:`ContractError`：数据不合规都抛它。
"""

from .config import (
    DEFAULT_CONFIG,
    MODEL_CHANNELS,
    REFERENCE_POLICIES,
    AppConfig,
    ConfigError,
    DecisionConfig,
    DisplayConfig,
    ProcessingConfig,
    RuntimeConfig,
    SourceConfig,
)
from .events import ReferenceMarker, ResetReason
from .frames import (
    CONTRACT_VERSION,
    EEG_UNIT,
    ContractError,
    EEGChunk,
    IncompatibleSourceError,
    SourceInfo,
    WindowRange,
)
from .outputs import (
    QUALITY_REASONS,
    DecisionFrame,
    DecisionState,
    QualityReport,
    Scores,
    utc_now_iso,
)
from .protocols import ChunkSequenceError, ChunkStreamGuard, DataValidator

__all__ = [
    "CONTRACT_VERSION",
    "DEFAULT_CONFIG",
    "MODEL_CHANNELS",
    "REFERENCE_POLICIES",
    "AppConfig",
    "ChunkSequenceError",
    "ChunkStreamGuard",
    "ConfigError",
    "ContractError",
    "DataValidator",
    "DecisionConfig",
    "DecisionFrame",
    "DecisionState",
    "DisplayConfig",
    "EEGChunk",
    "EEG_UNIT",
    "IncompatibleSourceError",
    "ProcessingConfig",
    "QUALITY_REASONS",
    "QualityReport",
    "ReferenceMarker",
    "ResetReason",
    "RuntimeConfig",
    "Scores",
    "SourceConfig",
    "SourceInfo",
    "WindowRange",
    "utc_now_iso",
]
