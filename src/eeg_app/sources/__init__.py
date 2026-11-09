"""数据源：EDF 读取（``input``）与 1 倍速回放（``replay``）。

分层只有两层：

* :mod:`eeg_app.sources.input`：文件 → :class:`~eeg_app.sources.input.Recording`
  （样本数组 + 档案卡 + 参考标注）。只管读，不管节奏。
* :mod:`eeg_app.sources.replay`：:class:`~eeg_app.sources.replay.ReplaySource`
  （分块 + 按源时间排程 + 暂停/继续/停止）。只管给，一次给一块。

在线路径只接收 ``EEGChunk`` 与 ``SourceInfo``；参考标注留在 ``Recording.markers`` 里，
不经过这里，也不进解码（PRD §5.3 / docs/01）。
"""

from .input import (
    SOURCE_ID,
    Recording,
    load_edf,
    load_recording,
    recording_id_of,
    recording_path,
    standardize_channel_names,
)
from .replay import ReplaySource, SourceState, chunk_samples_for, open_replay

__all__ = [
    "SOURCE_ID",
    "Recording",
    "ReplaySource",
    "SourceState",
    "chunk_samples_for",
    "load_edf",
    "load_recording",
    "open_replay",
    "recording_id_of",
    "recording_path",
    "standardize_channel_names",
]
