"""决策状态机：invalid > warming_up > uncertain / 已确认类别。"""

from __future__ import annotations

from eeg_core.contracts import (
    DEFAULT_CONFIG,
    MODEL_CHANNELS,
    DecisionConfig,
    DecisionState,
    ModelInputSpec,
    QualityReport,
    Scores,
    WindowRange,
)
from eeg_core.decoding.decision import DecisionEngine


def _spec() -> ModelInputSpec:
    return ModelInputSpec(
        model_id="unit-model",
        pipeline_version="unit-pipe",
        channels=MODEL_CHANNELS,
        sample_rate_hz=160.0,
        reference_policy="model_channel_average",
        bands_hz=((8.0, 13.0), (13.0, 30.0)),
        window_sec=2.0,
        step_sec=0.25,
        feature_layout="channel_band",
        feature_floor=1e-12,
        classes=("idle", "left_hand", "right_hand"),
        decision=DecisionConfig(),
        weights_sha256="0" * 64,
        preprocessing_hash="0" * 16,
    )


def _engine() -> DecisionEngine:
    return DecisionEngine(
        config=DEFAULT_CONFIG,
        spec=_spec(),
        session_id="unit-session",
        recording_id="S999R04",
    )


def _window() -> WindowRange:
    return WindowRange(sample_rate_hz=160.0, start_sample=480, end_sample=800)


def test_invalid_precedes_warmup_and_scores() -> None:
    engine = _engine()
    frame = engine.update(
        quality=QualityReport.failed(("amplitude",), bad_channels=("Fp1",)),
        ready=False,
        scores=None,
        window=None,
        emitted_monotonic_ms=0.0,
    )
    assert frame.decision_state is DecisionState.INVALID
    assert frame.scores is None
    assert frame.window is None
    assert frame.quality.valid is False


def test_warming_up_has_no_window_or_scores() -> None:
    engine = _engine()
    frame = engine.update(
        quality=QualityReport.ok(),
        ready=False,
        scores=None,
        window=None,
        emitted_monotonic_ms=0.0,
        source_t_sec=1.0,
    )
    assert frame.decision_state is DecisionState.WARMING_UP
    assert frame.window is None
    assert frame.scores is None
    assert frame.source_t_sec == 1.0


def test_low_score_is_uncertain_not_idle() -> None:
    engine = _engine()
    scores = Scores(idle=0.40, left_hand=0.35, right_hand=0.25)
    frame = engine.update(
        quality=QualityReport.ok(),
        ready=True,
        scores=scores,
        window=_window(),
        emitted_monotonic_ms=0.0,
    )
    assert frame.decision_state is DecisionState.UNCERTAIN
    assert frame.scores == scores
    assert frame.window is not None


def test_confirm_updates_then_action() -> None:
    engine = _engine()
    scores = Scores(idle=0.05, left_hand=0.90, right_hand=0.05)
    frames = [
        engine.update(
            quality=QualityReport.ok(),
            ready=True,
            scores=scores,
            window=_window(),
            emitted_monotonic_ms=index * 250.0,
        )
        for index in range(3)
    ]
    assert [frame.decision_state for frame in frames[:2]] == [
        DecisionState.UNCERTAIN,
        DecisionState.UNCERTAIN,
    ]
    assert frames[2].decision_state is DecisionState.LEFT_HAND
    assert frames[2].intent is DecisionState.LEFT_HAND


def test_confirmation_resets_after_low_score() -> None:
    engine = _engine()
    high = Scores(idle=0.05, left_hand=0.90, right_hand=0.05)
    low = Scores(idle=0.40, left_hand=0.35, right_hand=0.25)
    engine.update(
        quality=QualityReport.ok(),
        ready=True,
        scores=high,
        window=_window(),
        emitted_monotonic_ms=0.0,
    )
    engine.update(
        quality=QualityReport.ok(),
        ready=True,
        scores=low,
        window=_window(),
        emitted_monotonic_ms=250.0,
    )
    engine.update(
        quality=QualityReport.ok(),
        ready=True,
        scores=high,
        window=_window(),
        emitted_monotonic_ms=500.0,
    )
    frame = engine.update(
        quality=QualityReport.ok(),
        ready=True,
        scores=high,
        window=_window(),
        emitted_monotonic_ms=750.0,
    )
    assert frame.decision_state is DecisionState.UNCERTAIN


def test_invalid_with_window_keeps_evidence() -> None:
    engine = _engine()
    window = _window()
    frame = engine.update(
        quality=QualityReport.failed(("flatline",), bad_channels=("Cz",)),
        ready=True,
        scores=None,
        window=window,
        emitted_monotonic_ms=0.0,
    )
    assert frame.decision_state is DecisionState.INVALID
    assert frame.window == window
    assert frame.source_t_sec == window.source_t_sec


def test_frame_json_keys() -> None:
    engine = _engine()
    frame = None
    for index in range(3):
        frame = engine.update(
            quality=QualityReport.ok(),
            ready=True,
            scores=Scores(idle=0.9, left_hand=0.05, right_hand=0.05),
            window=_window(),
            emitted_monotonic_ms=index * 250.0,
        )
    assert frame is not None
    payload = frame.to_dict()
    assert payload["decision_state"] == "idle"
    assert payload["scores"] == {"idle": 0.9, "left_hand": 0.05, "right_hand": 0.05}
    assert payload["window_end_sample"] == 800
