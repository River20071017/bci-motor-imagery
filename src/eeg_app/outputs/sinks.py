"""有界后台存档队列；不依赖 Qt，也不改动冻结的 contracts。"""

from __future__ import annotations

import math
import queue
import threading
import time
from collections.abc import Mapping
from pathlib import Path

from eeg_core.contracts import DecisionFrame, IntentEvent, RuntimeEvent

from .archive import SessionArchive


class ArchiveWriteError(RuntimeError):
    """调用方必须停止供数/暂停会话，并向用户显示本错误。"""


class JsonlResultSink:
    """一个会话一个线程，只负责保存传入对象。

    submit 返回只表示入队；close 成功后才可显示“保存完成”。
    status 可由 C 的主线程轮询，后台不直接操作界面。磁盘失败/队列满后
    failed 会一直保持，后续 submit/close 都明确报错，不静默丢记录。
    """

    def __init__(
        self,
        root: str | Path,
        session_id: str,
        metadata: Mapping[str, object],
        *,
        queue_size: int = 256,
        flush_interval_sec: float = 1.0,
    ):
        if isinstance(queue_size, bool) or not isinstance(queue_size, int) or queue_size < 1:
            raise ValueError("queue_size 必须是正整数")
        if not math.isfinite(flush_interval_sec) or not 0 < flush_interval_sec <= 1:
            raise ValueError("flush_interval_sec 必须在 (0, 1] 秒内")
        self._lock = threading.RLock()
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._closing = threading.Event()
        self._ready = threading.Event()
        self._error: str | None = None
        self._state = "opening"
        self._completed = False
        self._reason = "stopped"
        self._directory: Path | None = None
        self._interval = flush_interval_sec
        # 在线程中完成目录和文件 IO；构造函数仅等待初始化结果。
        self._worker = threading.Thread(
            target=self._run,
            args=(root, session_id, dict(metadata)),
            name=f"archive-{session_id}",
            daemon=True,
        )
        self._worker.start()
        self._ready.wait()
        self.check_error()

    @property
    def status(self) -> dict[str, object]:
        with self._lock:
            return {
                "state": self._state,
                "directory": str(self._directory) if self._directory else None,
                "error": self._error,
                "queued": self._queue.qsize(),
                "completed": self._completed and self._state == "closed",
            }

    def check_error(self) -> None:
        with self._lock:
            if self._error is not None:
                raise ArchiveWriteError(self._error)

    def _fail(self, error: Exception) -> None:
        with self._lock:
            if self._error is None:
                self._error = f"{type(error).__name__}: {error}"
            self._state = "failed"
            self._closing.set()

    def submit(self, item: DecisionFrame | IntentEvent | RuntimeEvent) -> None:
        if not isinstance(item, (DecisionFrame, IntentEvent, RuntimeEvent)):
            raise TypeError("只能存档 DecisionFrame / IntentEvent / RuntimeEvent")
        with self._lock:
            self.check_error()
            if self._state != "writing":
                raise ArchiveWriteError("会话已关闭或正在关闭，不能继续提交")
            try:
                self._queue.put_nowait(item)
            except queue.Full as error:
                self._fail(ArchiveWriteError("存档队列已满，必须暂停会话"))
                raise ArchiveWriteError(self._error) from error

    def close(
        self, *, completed: bool = True, reason: str = "finished", timeout: float = 10
    ) -> None:
        if not isinstance(completed, bool) or not isinstance(reason, str) or not reason.strip():
            raise ValueError("completed 必须为 bool，reason 必须是非空字符串")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout 必须为有限正数")
        with self._lock:
            if not self._closing.is_set():
                self._completed = completed
                self._reason = reason
                self._state = "closing"
                self._closing.set()
        self._worker.join(timeout)
        if self._worker.is_alive():
            self._fail(ArchiveWriteError("等待存档排空超时，会话未确认保存完成"))
        self.check_error()

    def _run(self, root, session_id, metadata) -> None:
        archive = None
        try:
            archive = SessionArchive(root, session_id, metadata)
            with self._lock:
                self._directory = archive.directory
                self._state = "writing"
            self._ready.set()
            deadline = time.monotonic() + self._interval
            while True:
                if self._closing.is_set() and (self._error is not None or self._queue.empty()):
                    break
                try:
                    item = self._queue.get(
                        timeout=max(0.001, min(0.05, deadline - time.monotonic()))
                    )
                except queue.Empty:
                    item = None
                if item is not None:
                    try:
                        if isinstance(item, DecisionFrame):
                            archive.append_decision(item)
                        elif isinstance(item, IntentEvent):
                            archive.append_event(item)
                        else:
                            archive.append_runtime(item)
                    finally:
                        self._queue.task_done()
                if time.monotonic() >= deadline:
                    archive.flush()
                    deadline = time.monotonic() + self._interval
            with self._lock:
                completed = self._completed and self._error is None
                reason = self._error or self._reason
            archive.close(completed=completed, reason=reason)
            with self._lock:
                if self._error is None:
                    self._state = "closed"
        except Exception as error:
            self._fail(error)
            if archive is not None:
                try:
                    archive.close(completed=False, reason=self._error)
                except Exception:
                    # 磁盘已不可写时，内存错误状态和初始 completed=false 是最后证据。
                    pass
        finally:
            self._ready.set()
