"""数据源：统一 MNE 读取层（``input``）与 1 倍速回放（``replay``）。

分层只有两层：

* :mod:`eeg_app.sources.input`：文件 → :class:`~eeg_app.sources.input.Recording`
  （样本数组 + 档案卡 + 参考标注）。格式靠 MNE 按扩展名分派，通道名靠别名表，
  重采样是显式可选项；只管读，不管节奏。
* :mod:`eeg_app.sources.replay`：:class:`~eeg_app.sources.replay.ReplaySource`
  （分块 + 按源时间排程 + 暂停/继续/停止）。只管给，一次给一块。

在线路径只接收 ``EEGChunk`` 与 ``SourceInfo``；参考标注留在 ``Recording.markers`` 里，
不经过这里，也不进解码（PRD §5.3 / docs/01）。
"""

from .bnci2014 import (
    BNCI2014_SOURCE_ID,
    SEALED_SESSIONS,
    load_bnci2014_session,
    parse_bnci2014_mat,
)
from .input import (
    SEALED_RUNS,
    SOURCE_ID,
    Recording,
    apply_channel_aliases,
    load_edf,
    load_file,
    load_recording,
    read_raw,
    recording_from_raw,
    recording_id_of,
    recording_path,
    standardize_channel_names,
)
from .replay import ReplaySource, SourceState, chunk_samples_for, open_replay

__all__ = [
    "BNCI2014_SOURCE_ID",
    "SEALED_RUNS",
    "SEALED_SESSIONS",
    "SOURCE_ID",
    "Recording",
    "ReplaySource",
    "SourceState",
    "apply_channel_aliases",
    "chunk_samples_for",
    "load_bnci2014_session",
    "load_edf",
    "load_file",
    "load_recording",
    "open_replay",
    "parse_bnci2014_mat",
    "read_raw",
    "recording_from_raw",
    "recording_id_of",
    "recording_path",
    "standardize_channel_names",
]
