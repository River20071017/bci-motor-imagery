"""契约 1.0：模块之间唯一的接缝定义。

本包只放数据结构与协议：不读文件、不 import mne / PySide6，也不含业务判断。字段或语义
变化必须先升级 :data:`CONTRACT_VERSION`，同步 ``docs/01_接口约定_1.0.md`` 与样例，再分别
改双方模块（README「开发规范」）。

当前落地的是输入侧与中间层契约：``EEGChunk``（六组信息：身份、时间、形状、物理属性、
通道、流级事件）、``WindowRange``（观察窗网格）、``QualityReport``（质量结论）、
``SourceInfo`` / ``DataSource``、``ChunkStreamGuard``、``ResetReason`` /
``StreamConsumer``（重置语义）、参考轨专用的 ``ReferenceMarker`` / ``MarkerSource``，
以及全局参数唯一的出处 ``AppConfig``。
``DecisionFrame``、会话状态与动作事件按同一版本号在后续提交中补齐。
"""

from .config import (
    DEFAULT_CONFIG,
    MODEL_CHANNELS,
    AppConfig,
    ConfigError,
    DecisionConfig,
    DisplayConfig,
    ProcessingConfig,
    RuntimeConfig,
    SourceConfig,
)
from .events import ReferenceMarker
from .frames import (
    CONTRACT_VERSION,
    EEG_UNIT,
    QUALITY_REASONS,
    DecisionFrame,
    DecisionState,
    EEGChunk,
    EEGChunkError,
    QualityReport,
    Scores,
    WindowRange,
    utc_now_iso,
)
from .protocols import (
    ChunkSequenceError,
    ChunkStreamGuard,
    DataSource,
    IncompatibleSourceError,
    MarkerSource,
    ResetReason,
    SourceClosedError,
    SourceError,
    SourceInfo,
    StreamConsumer,
)

__all__ = [
    "CONTRACT_VERSION",
    "DEFAULT_CONFIG",
    "MODEL_CHANNELS",
    "AppConfig",
    "ChunkSequenceError",
    "ChunkStreamGuard",
    "ConfigError",
    "DataSource",
    "DecisionConfig",
    "DecisionFrame",
    "DecisionState",
    "DisplayConfig",
    "EEGChunk",
    "EEGChunkError",
    "EEG_UNIT",
    "IncompatibleSourceError",
    "MarkerSource",
    "ProcessingConfig",
    "QUALITY_REASONS",
    "QualityReport",
    "ReferenceMarker",
    "ResetReason",
    "RuntimeConfig",
    "Scores",
    "SourceClosedError",
    "SourceConfig",
    "SourceError",
    "SourceInfo",
    "StreamConsumer",
    "WindowRange",
    "utc_now_iso",
]
