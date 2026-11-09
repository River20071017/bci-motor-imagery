"""D：会话输出。这里只导出具体实现，不新增冻结契约的 Protocol。"""

from .sinks import ArchiveWriteError, JsonlResultSink

__all__ = ["ArchiveWriteError", "JsonlResultSink"]
