"""A 的会话接线验收。使用合成对象和 D 的真实 writer/reader。"""

from __future__ import annotations

import csv
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, get_ident
from time import monotonic
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from eeg_app.app import start_archived_session
from eeg_app.outputs import JsonlResultSink
from eeg_app.outputs.archive import SessionArchive
from eeg_app.sources import Recording, ReplaySource, SourceState
from eeg_core.contracts import (
    DEFAULT_CONFIG,
    ContractError,
    DecisionState,
    IntentEvent,
    ModelInputSpec,
    QualityReport,
    RuntimeEvent,
    RuntimeEventKind,
    Scores,
    SessionState,
    SourceInfo,
    WindowRange,
)
from eeg_core.version import PIPELINE_VERSION
from examples.read_session import inspect_session


def _frame(engine, state):
    end = 480 + 40 * engine.seq
    valid = state is not DecisionState.INVALID
    ready = state is not DecisionState.WARMING_UP
    scores = {
        DecisionState.IDLE: Scores(0.8, 0.1, 0.1),
        DecisionState.LEFT_HAND: Scores(0.1, 0.8, 0.1),
        DecisionState.RIGHT_HAND: Scores(0.1, 0.1, 0.8),
        DecisionState.UNCERTAIN: Scores(0.4, 0.3, 0.3),
    }.get(state)
    return engine.update(
        quality=QualityReport.ok() if valid else QualityReport.failed(("non_finite",)),
        ready=ready,
        scores=scores,
        window=WindowRange(160, end - 320, end) if valid and ready else None,
        source_t_sec=end / 160,
        emitted_monotonic_ms=monotonic() * 1000,
    )


def _intent(first, last, reason):
    return IntentEvent(
        event_id=f"{first.session_id}/evt-{first.seq}",
        session_id=first.session_id,
        recording_id=first.recording_id,
        state=first.decision_state,
        start_sample=first.window.end_sample,
        start_decision_seq=first.seq,
        end_sample=last.window.end_sample,
        end_decision_seq=last.seq,
        end_reason=reason,
    )


def _json(directory, filename):
    return json.loads((directory / filename).read_text(encoding="utf-8"))


def _jsonl(directory, filename):
    return [
        json.loads(line)
        for line in (directory / filename).read_text(encoding="utf-8").splitlines()
    ]


class TestSessionArchive(TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory(dir=Path.cwd())))
        self.executor = self.enterContext(ThreadPoolExecutor(max_workers=1))
        channels = DEFAULT_CONFIG.source.model_channels
        self.config = replace(
            DEFAULT_CONFIG,
            source=replace(DEFAULT_CONFIG.source, channel_count=len(channels)),
            decision=replace(DEFAULT_CONFIG.decision, confirm_updates=1),
        )
        source = self.root / "synthetic-input.bin"
        source.write_bytes(b"archive provenance fixture; no EEG recording")
        self.recording = Recording(
            SourceInfo("synthetic", "format-fixture", 160, channels),
            np.zeros((len(channels), 800)),
            path=source,
            sha256=sha256(source.read_bytes()).hexdigest(),
        )
        self.model = ModelInputSpec(
            model_id="fixture-model",
            pipeline_version=PIPELINE_VERSION,
            channels=channels,
            sample_rate_hz=160,
            reference_policy=self.config.source.reference_policy,
            bands_hz=self.config.processing.bands_hz,
            window_sec=2,
            step_sec=0.25,
            feature_layout="channel_band",
            feature_floor=self.config.processing.log_variance_floor,
            classes=("idle", "left_hand", "right_hand"),
            decision=self.config.decision,
            weights_sha256=sha256(b"synthetic model identity").hexdigest(),
            preprocessing_hash=self.config.preprocessing_hash,
        )
        self.frames = []
        self.errors = []
        self.stopped = Event()

    def _start(self, produce, **overrides):
        options = {
            "output_root": self.root,
            "recording": self.recording,
            "model_spec": self.model,
            "config": self.config,
            "produce": produce,
            "stop_production": self.stopped.set,
            "close_intent": lambda _reason: None,
            "emit_decision": self.frames.append,
            "emit_archive_error": lambda session_id, error: self.errors.append((session_id, error)),
            "app_version": "0.1.0",
        }
        return start_archived_session(self.executor, **(options | overrides))

    def test_eof_archives_exact_ui_frames_and_closes_active_intent(self):
        frames = []
        io_calls = []
        states = (
            DecisionState.WARMING_UP,
            DecisionState.INVALID,
            DecisionState.UNCERTAIN,
            DecisionState.IDLE,
            DecisionState.LEFT_HAND,
            DecisionState.RIGHT_HAND,
            DecisionState.RIGHT_HAND,
        )

        class TracedSink(JsonlResultSink):
            def __init__(self, *args, **kwargs):
                io_calls.append(("open", get_ident()))
                super().__init__(*args, **kwargs)

            def close(self, **kwargs):
                io_calls.append(("close", get_ident()))
                super().close(**kwargs)

        def produce(archive, engine):
            archive.submit(
                RuntimeEvent(archive.session_id, monotonic() * 1000, RuntimeEventKind.START)
            )
            for state in states:
                frame = _frame(engine, state)
                frames.append(frame)
                archive.submit(frame)
            archive.submit(_intent(frames[4], frames[5], "state_changed"))
            return "finished"

        def close_intent(reason):
            self.assertTrue(self.stopped.is_set())
            return _intent(frames[5], frames[6], reason)

        with patch("eeg_app.runtime.session.JsonlResultSink", TracedSink):
            session, work = self._start(produce, close_intent=close_intent)
            self.assertEqual(work.result(timeout=5), SessionState.FINISHED)

        self.assertEqual([phase for phase, _ in io_calls], ["open", "close"])
        self.assertEqual(io_calls[0][1], io_calls[1][1])
        self.assertNotEqual(io_calls[0][1], get_ident())
        self.assertEqual(len(self.frames), len(frames))
        self.assertTrue(all(a is b for a, b in zip(self.frames, frames, strict=True)))
        directory = Path(session.status["directory"])
        self.assertTrue(session.status["completed"])
        self.assertEqual(inspect_session(directory)["counts"], {
            "decisions": 7, "events": 2, "runtime": 3,
        })
        saved = _jsonl(directory, "decisions.jsonl")
        self.assertEqual(saved, [frame.to_dict() for frame in frames])
        self.assertEqual({row["decision_state"] for row in saved}, {s.value for s in DecisionState})
        manifest = _json(directory, "manifest.json")["metadata"]
        self.assertEqual(manifest["source_files"], [{
            "path": str(self.recording.path), "sha256": self.recording.sha256,
        }])
        self.assertEqual(manifest["config"], self.config.to_dict())
        self.assertEqual(manifest["app_version"], "0.1.0")
        self.assertIn("python", manifest["dependencies"])
        self.assertEqual(_json(directory, "summary.json")["end_reason"], "finished")
        with (directory / "events.csv").open(encoding="utf-8", newline="") as stream:
            self.assertEqual([row["end_reason"] for row in csv.DictReader(stream)], [
                "state_changed", "finished",
            ])

    def test_user_stop_completes_archive_and_next_session_restarts_seq(self):
        source = ReplaySource(self.recording, config=self.config)
        waiting = Event()
        frames = []

        def produce(archive, engine):
            source.start()
            source.pause()
            for _ in range(2):
                frame = _frame(engine, DecisionState.LEFT_HAND)
                frames.append(frame)
                archive.submit(frame)
            waiting.set()
            self.assertIsNone(source.read_chunk())
            return "stopped"

        first, work = self._start(
            produce,
            stop_production=source.stop,
            close_intent=lambda reason: _intent(frames[0], frames[-1], reason),
        )
        try:
            self.assertTrue(waiting.wait(5))
            source.stop()
            self.assertEqual(work.result(timeout=5), SessionState.STOPPED)
        finally:
            source.stop()
        first_dir = Path(first.status["directory"])
        summary = _json(first_dir, "summary.json")
        self.assertTrue(summary["completed"])
        self.assertEqual(summary["end_reason"], "stopped")
        self.assertEqual(_jsonl(first_dir, "runtime.jsonl")[-1]["kind"], "stop")
        self.assertEqual(inspect_session(first_dir)["counts"]["events"], 1)

        recording = replace(
            self.recording,
            info=replace(self.recording.info, recording_id="other-recording"),
            path=None,
            sha256=None,
        )
        config = replace(self.config, decision=replace(self.config.decision, threshold=0.70))
        model = replace(
            self.model,
            model_id="other-model",
            decision=config.decision,
            preprocessing_hash=config.preprocessing_hash,
        )

        def produce_next(archive, engine):
            with self.assertRaisesRegex(ContractError, "当前会话"):
                archive.submit(frames[0])
            archive.submit(_frame(engine, DecisionState.WARMING_UP))
            return "finished"

        second, work = self._start(
            produce_next, recording=recording, model_spec=model, config=config
        )
        self.assertEqual(work.result(timeout=5), SessionState.FINISHED)
        second_dir = Path(second.status["directory"])
        self.assertNotEqual(first.session_id, second.session_id)
        self.assertNotEqual(first_dir, second_dir)
        saved = _jsonl(second_dir, "decisions.jsonl")
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0]["seq"], 0)
        self.assertEqual(saved[0]["model_id"], model.model_id)
        self.assertEqual(saved[0]["recording_id"], recording.info.recording_id)
        self.assertEqual(saved[0]["config_hash"], config.config_hash)
        self.assertEqual(_json(second_dir, "manifest.json")["metadata"]["source_files"], [])
        self.assertEqual(_json(first_dir, "summary.json"), summary)

    def test_async_write_failure_stops_paused_source_without_more_decisions(self):
        source = ReplaySource(self.recording, config=self.config)
        waiting = Event()

        def disk_error(_archive, _frame):
            if not waiting.wait(5):
                raise TimeoutError("测试生产循环未进入暂停")
            raise OSError("simulated disk full")

        def produce(archive, engine):
            source.start()
            source.pause()
            archive.submit(_frame(engine, DecisionState.WARMING_UP))
            waiting.set()
            self.assertIsNone(source.read_chunk())
            return "stopped"

        with patch.object(SessionArchive, "append_decision", disk_error):
            session, work = self._start(produce, stop_production=source.stop)
            try:
                self.assertEqual(work.result(timeout=5), SessionState.ERROR)
            finally:
                source.stop()
        self.assertEqual(source.state, SourceState.STOPPED)
        self._assert_failed(session, "simulated disk full")
        self.assertFalse(_json(Path(session.status["directory"]), "summary.json")["completed"])

    def test_full_queue_fails_session_and_does_not_publish_rejected_frame(self):
        start_written, writer_waiting, release_writer = Event(), Event(), Event()
        append_runtime = SessionArchive.append_runtime
        append_decision = SessionArchive.append_decision

        class SmallSink(JsonlResultSink):
            def __init__(self, *args):
                super().__init__(*args, queue_size=1)

        def write_runtime(archive, event):
            append_runtime(archive, event)
            start_written.set()

        def block_writer(archive, frame):
            writer_waiting.set()
            if not release_writer.wait(5):
                raise TimeoutError("测试未收到队列满通知")
            append_decision(archive, frame)

        def produce(archive, engine):
            self.assertTrue(start_written.wait(5))
            archive.submit(_frame(engine, DecisionState.WARMING_UP))
            self.assertTrue(writer_waiting.wait(5))
            archive.submit(_frame(engine, DecisionState.WARMING_UP))
            archive.submit(_frame(engine, DecisionState.WARMING_UP))
            self.fail("队列已满时提交应失败")

        def error_signal(session_id, error):
            self.errors.append((session_id, error))
            release_writer.set()

        with (
            patch("eeg_app.runtime.session.JsonlResultSink", SmallSink),
            patch.object(SessionArchive, "append_runtime", write_runtime),
            patch.object(SessionArchive, "append_decision", block_writer),
        ):
            session, work = self._start(produce, emit_archive_error=error_signal)
            try:
                self.assertEqual(work.result(timeout=5), SessionState.ERROR)
            finally:
                release_writer.set()
        self.assertTrue(self.stopped.is_set())
        self.assertEqual([frame.seq for frame in self.frames], [0, 1])
        self._assert_failed(session, "队列已满")

    def test_open_failure_prevents_production(self):
        invalid_root = self.root / "file-as-directory"
        invalid_root.touch()
        session, work = self._start(
            lambda *_: self.fail("初始化失败后不能供数"), output_root=invalid_root
        )
        self.assertEqual(work.result(timeout=5), SessionState.ERROR)
        self.assertTrue(self.stopped.is_set())
        self._assert_failed(session)

    def test_close_failure_reports_error_instead_of_finished(self):
        close = SessionArchive.close

        def fail_at_close(archive, *, completed, reason):
            if completed:
                raise OSError("simulated final flush failure")
            close(archive, completed=completed, reason=reason)

        with patch.object(SessionArchive, "close", fail_at_close):
            session, work = self._start(lambda *_: "finished")
            self.assertEqual(work.result(timeout=5), SessionState.ERROR)
        self.assertTrue(self.stopped.is_set())
        self._assert_failed(session, "simulated final flush failure")
        self.assertFalse(_json(Path(session.status["directory"]), "summary.json")["completed"])

    def _assert_failed(self, session, message=None):
        status = session.status
        self.assertEqual(status["state"], "failed")
        self.assertFalse(status["completed"])
        self.assertEqual(self.errors, [(session.session_id, status["error"])])
        if message is not None:
            self.assertIn(message, status["error"])
