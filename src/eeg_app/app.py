"""A 的应用组装。存档入口可接到会话生产循环，供 GUI 异步启动。"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from importlib import metadata as package_metadata
from pathlib import Path
from platform import python_version
from uuid import uuid4

from eeg_core.contracts import AppConfig, DecisionFrame, IntentEvent, ModelInputSpec, SessionState
from eeg_core.decoding import DecisionEngine

from .runtime.session import ArchivedSession, EndReason
from .sources import Recording

_DEPENDENCIES = ("numpy", "scipy", "mne", "scikit-learn", "PySide6", "pyqtgraph")


def _archive_metadata(
    recording: Recording, spec: ModelInputSpec, config: AppConfig, app_version: str | None
) -> dict[str, object]:
    dependencies = {"python": python_version()}
    for name in _DEPENDENCIES:
        try:
            dependencies[name] = package_metadata.version(name)
        except package_metadata.PackageNotFoundError:
            continue

    source_files = []
    if recording.path is not None:
        source_file = {"path": str(recording.path)}
        if recording.sha256 is not None:
            source_file["sha256"] = recording.sha256
        source_files.append(source_file)

    metadata = {
        "recording_id": recording.info.recording_id,
        "model_id": spec.model_id,
        "pipeline_version": spec.pipeline_version,
        "config_hash": config.config_hash,
        "config": config.to_dict(),
        "dependencies": dependencies,
        "source_files": source_files,
    }
    if app_version is None:
        try:
            app_version = package_metadata.version("bci-motor-imagery")
        except package_metadata.PackageNotFoundError:
            pass
    if app_version is not None:
        metadata["app_version"] = app_version
    return metadata


def start_archived_session(
    executor: ThreadPoolExecutor,
    *,
    output_root: str | Path,
    recording: Recording,
    model_spec: ModelInputSpec,
    config: AppConfig,
    produce: Callable[[ArchivedSession, DecisionEngine], EndReason],
    stop_production: Callable[[], None],
    close_intent: Callable[[EndReason], IntentEvent | None],
    emit_decision: Callable[[DecisionFrame], None],
    emit_archive_error: Callable[[str, str], None],
    app_version: str | None = None,
) -> tuple[ArchivedSession, Future[SessionState]]:
    """为一条已加载的记录和模型开启新会话，立即返回状态句柄和工作 Future。

    真实 EDF 使用 ``load_edf(..., with_sha256=True)`` 的 Recording。源码或打包
    入口可显式传实际 app_version；没有可用版本或文件哈希时省略未知字段。
    GUI 使用排队信号接收决策和错误，并轮询 session.status / future.done()。
    停止操作请求源与生产循环退出，随后由工作线程排空归档。
    """
    model_spec.check_compatible(recording.info, config)
    session = ArchivedSession(
        output_root,
        uuid4().hex,
        stop_production=stop_production,
        emit_decision=emit_decision,
        emit_archive_error=emit_archive_error,
    )

    def run() -> SessionState:
        engine = DecisionEngine(
            config=config,
            spec=model_spec,
            session_id=session.session_id,
            recording_id=recording.info.recording_id,
        )
        return session.run(
            _archive_metadata(recording, model_spec, config, app_version),
            lambda archive: produce(archive, engine),
            close_intent,
        )

    return session, executor.submit(run)
