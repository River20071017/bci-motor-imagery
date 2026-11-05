"""契约单测：把字段、单位、枚举和序列化格式钉死（冻结前的那把“门锁”）。

只依赖 numpy 与契约本身，不读数据、不需要 mne。

    python -m pytest tests/unit/core/test_contracts.py
    python tests/unit/core/test_contracts.py        # 没有 pytest 也能跑
"""

from __future__ import annotations

import json
import sys
from dataclasses import FrozenInstanceError, replace

import numpy as np

import eeg_core.contracts as contracts
from eeg_core.contracts import (
    DEFAULT_CONFIG,
    END_REASONS,
    FEATURE_LAYOUTS,
    MODEL_CHANNELS,
    QUALITY_REASONS,
    REFERENCE_POLICIES,
    ChunkSequenceError,
    ChunkStreamGuard,
    ConfigError,
    ContractError,
    DecisionFrame,
    DecisionState,
    EEGChunk,
    IncompatibleModelError,
    IncompatibleSourceError,
    IntentEvent,
    ModelInputSpec,
    QualityReport,
    ReferenceMarker,
    ResetReason,
    RuntimeEvent,
    RuntimeEventKind,
    Scores,
    SessionState,
    SourceInfo,
    WindowRange,
    utc_now_iso,
)

CHANNELS = ("FC3", "FCz", "FC4", "C3", "Cz", "C4", "CP3", "CPz", "CP4")
SAMPLE_RATE = 160.0


def _expect(error_type: type[BaseException], call, label: str) -> None:
    try:
        call()
    except error_type:
        return
    except Exception as exc:  # noqa: BLE001 - 自检要报告实际异常
        raise AssertionError(
            f"{label}：期望 {error_type.__name__}，实际 {type(exc).__name__}: {exc}"
        ) from exc
    raise AssertionError(f"{label}：期望 {error_type.__name__}，但没有抛异常")


def _assign(mapping, key, value) -> None:
    mapping[key] = value


def _info(**overrides) -> SourceInfo:
    payload = {
        "source_id": "physionet-eegmmidb",
        "recording_id": "S001R04",
        "sample_rate_hz": SAMPLE_RATE,
        "channel_names": CHANNELS,
        "n_samples": 20000,
    }
    payload.update(overrides)
    return SourceInfo(**payload)


def _chunk(
    seq: int = 0, start_sample: int = 0, discontinuity: bool = False, **overrides
) -> EEGChunk:
    payload = {
        "source_id": "physionet-eegmmidb",
        "recording_id": "S001R04",
        "seq": seq,
        "start_sample": start_sample,
        "sample_rate_hz": SAMPLE_RATE,
        "channel_names": CHANNELS,
        "samples": np.zeros((len(CHANNELS), 8)),
        "released_monotonic_ms": 1000.0 + seq,
        "discontinuity": discontinuity,
    }
    payload.update(overrides)
    return EEGChunk(**payload)


def _window(index: int = 3) -> WindowRange:
    return WindowRange.from_grid_index(
        index, sample_rate_hz=SAMPLE_RATE, window_samples=320, hop_samples=40
    )


def _frame(**overrides) -> DecisionFrame:
    window = overrides.pop("window", _window(12))
    payload = {
        "session_id": "demo-20261103-001",
        "recording_id": "S001R04",
        "seq": 12,
        "decision_state": DecisionState.LEFT_HAND,
        "intent": DecisionState.LEFT_HAND,
        "sample_rate_hz": SAMPLE_RATE,
        "source_t_sec": window.source_t_sec,
        "generated_at_utc": utc_now_iso(),
        "emitted_monotonic_ms": 12345.6,
        "model_id": "mi-S001-lda-20261103",
        "pipeline_version": "pipe-1.0",
        "config_hash": DEFAULT_CONFIG.config_hash,
        "window": window,
        "scores": Scores(idle=0.12, left_hand=0.71, right_hand=0.17),
    }
    payload.update(overrides)
    return DecisionFrame(**payload)


def _model_spec(**overrides) -> ModelInputSpec:
    payload = {
        "model_id": "mi-S001-lda-20261103",
        "pipeline_version": "pipe-1.0",
        "channels": MODEL_CHANNELS,
        "sample_rate_hz": SAMPLE_RATE,
        "reference_policy": DEFAULT_CONFIG.source.reference_policy,
        "bands_hz": DEFAULT_CONFIG.processing.bands_hz,
        "window_sec": DEFAULT_CONFIG.processing.window_sec,
        "step_sec": DEFAULT_CONFIG.processing.step_sec,
        "feature_layout": "channel_band",
        "feature_floor": DEFAULT_CONFIG.processing.log_variance_floor,
        "classes": ("idle", "left_hand", "right_hand"),
        "decision": DEFAULT_CONFIG.decision,
        "weights_sha256": "a" * 64,
        "preprocessing_hash": DEFAULT_CONFIG.preprocessing_hash,
        "training_files": {"data/eegmmidb/S001/S001R04.edf": "b" * 64},
    }
    payload.update(overrides)
    return ModelInputSpec(**payload)


# --------------------------------------------------------------------- 导出面


def test_export_surface_is_pinned() -> None:
    """导出名单是契约的一部分：少一个、多一个都要显式改测试。"""
    assert len(contracts.__all__) == 36
    required = {
        "CONTRACT_VERSION",
        "EEGChunk",
        "WindowRange",
        "SourceInfo",
        "ReferenceMarker",
        "ResetReason",
        "SessionState",
        "RuntimeEvent",
        "RuntimeEventKind",
        "IntentEvent",
        "QualityReport",
        "DecisionState",
        "Scores",
        "DecisionFrame",
        "ModelInputSpec",
        "IncompatibleModelError",
        "ChunkStreamGuard",
        "AppConfig",
        "MODEL_CHANNELS",
        "REFERENCE_POLICIES",
        "FEATURE_LAYOUTS",
    }
    assert required <= set(contracts.__all__)


# --------------------------------------------------------------------- 输入侧


def test_chunk_shape_unit_and_derived_fields() -> None:
    chunk = _chunk()
    assert chunk.samples.shape == (9, 8)
    assert chunk.samples.dtype == np.float64
    assert chunk.unit == "V"
    assert chunk.n_channels == 9 and chunk.n_samples == 8
    assert chunk.end_sample == 8
    assert chunk.duration_sec == 0.05
    assert chunk.start_t_sec == 0.0
    assert chunk.source_t_sec == 0.05
    assert chunk.chunk_id == "S001R04/000000"
    assert chunk.channel_index("Cz") == 4


def test_chunk_samples_are_read_only_and_object_is_frozen() -> None:
    chunk = _chunk()
    assert not chunk.samples.flags.writeable
    _expect(FrozenInstanceError, lambda: setattr(chunk, "seq", 9), "块本身应不可变")


def test_chunk_rejects_bad_shapes_and_dtypes() -> None:
    _expect(ContractError, lambda: _chunk(samples=np.zeros(8)), "一维数组")
    _expect(ContractError, lambda: _chunk(samples=np.zeros((8, 8))), "通道数不符")
    _expect(ContractError, lambda: _chunk(samples=np.zeros((9, 8), dtype=np.int16)), "整数数组")
    _expect(ContractError, lambda: _chunk(samples=np.zeros((9, 0))), "零样本")


def test_chunk_allows_nan_but_reports_it() -> None:
    samples = np.zeros((9, 8))
    samples[0, 0] = np.nan
    chunk = _chunk(samples=samples)
    assert chunk.has_non_finite() is True


def test_chunk_metadata_keys_are_stable() -> None:
    assert set(_chunk().metadata()) == {
        "schema_version",
        "source_id",
        "recording_id",
        "seq",
        "start_sample",
        "sample_rate_hz",
        "channel_names",
        "channel_types",
        "released_monotonic_ms",
        "discontinuity",
    }


def test_guard_requires_new_record_boundary_on_first_chunk() -> None:
    guard = ChunkStreamGuard(_info())
    _expect(
        ChunkSequenceError,
        lambda: guard.validate(_chunk(seq=0, start_sample=0, discontinuity=False)),
        "首块必须标 discontinuity",
    )

    guard = ChunkStreamGuard(_info())
    guard.validate(_chunk(seq=0, start_sample=0, discontinuity=True))
    _expect(
        ChunkSequenceError,
        lambda: guard.validate(_chunk(seq=1, start_sample=16, discontinuity=False)),
        "块起点应接上一块末尾",
    )


def test_window_grid_is_anchored_at_zero() -> None:
    window = _window(3)
    assert (window.start_sample, window.end_sample) == (120, 440)
    assert window.n_samples == 320
    assert window.is_aligned(40) is True
    assert window.is_complete(440) is True
    assert window.is_complete(439) is False
    assert window.overlaps(WindowRange(SAMPLE_RATE, 400, 720)) is True
    assert window.overlaps(WindowRange(SAMPLE_RATE, 440, 760)) is False  # 首尾相接不算重叠
    _expect(ContractError, lambda: WindowRange(SAMPLE_RATE, 100, 100), "空窗")


def test_source_info_new_fields_are_read_only() -> None:
    info = _info(
        channel_positions={"C3": (1.0, 2.0, 3.0)},
        device_metadata={"manufacturer": "ACME", "hardware_filters": "0.5-100 Hz"},
    )
    assert info.channel_positions["C3"] == (1.0, 2.0, 3.0)
    assert info.device_metadata["manufacturer"] == "ACME"
    _expect(TypeError, lambda: _assign(info.channel_positions, "C3", (0, 0, 0)), "坐标只读")
    _expect(
        ContractError,
        lambda: _info(channel_positions={"XX": (0.0, 0.0, 0.0)}),
        "坐标键必须在通道表内",
    )
    _expect(
        ContractError,
        lambda: _info(channel_positions={"C3": (0.0, 0.0)}),
        "坐标必须是三元组",
    )


def test_check_compatible_reports_all_reasons() -> None:
    info = SourceInfo(
        source_id="physionet-eegmmidb",
        recording_id="S099R04",
        sample_rate_hz=250.0,
        channel_names=("C3", "C4"),
    )
    try:
        info.check_compatible(required_channels=MODEL_CHANNELS, sample_rate_hz=SAMPLE_RATE)
    except IncompatibleSourceError as error:
        assert len(error.reasons) == 2, error.reasons
    else:
        raise AssertionError("缺通道 + 采样率不符必须一次报全")


def test_reference_marker_keeps_raw_label_and_unknown_end() -> None:
    marker = ReferenceMarker(recording_id="S001R04", onset_sample=672, label="T1")
    assert marker.end_sample is None
    assert marker.covers_sample(700) is None
    assert marker.onset_t_sec(SAMPLE_RATE) == 4.2


# --------------------------------------------------------------------- 输出侧


def test_quality_report_mutual_exclusion() -> None:
    _expect(
        ContractError,
        lambda: QualityReport(valid=True, reasons=("non_finite",)),
        "合格不能带原因",
    )
    _expect(ContractError, lambda: QualityReport(valid=False), "不合格必须给原因")
    _expect(
        ContractError,
        lambda: QualityReport(valid=False, reasons=("oops",)),
        "原因码必须合法",
    )
    assert QualityReport.failed(("flatline",)).valid is False


def test_decision_frame_state_rules() -> None:
    _expect(
        ContractError,
        lambda: _frame(decision_state=DecisionState.WARMING_UP, intent=None, scores=None),
        "warming_up 不能带窗口",
    )
    _expect(
        ContractError,
        lambda: _frame(decision_state=DecisionState.LEFT_HAND, intent=DecisionState.RIGHT_HAND),
        "intent 必须等于状态",
    )
    _expect(
        ContractError,
        lambda: _frame(decision_state=DecisionState.INVALID, intent=None, scores=None),
        "invalid 必须带不合格质量",
    )
    _expect(
        ContractError,
        lambda: _frame(source_t_sec=99.0),
        "source_t_sec 必须等于窗口末端",
    )


def test_decision_frame_json_contract() -> None:
    frame = _frame()
    payload = json.loads(frame.to_json())
    assert payload["decision_id"] == "demo-20261103-001/000012"
    assert payload["decision_state"] == "left_hand"
    assert payload["scores"] == {"idle": 0.12, "left_hand": 0.71, "right_hand": 0.17}
    assert payload["quality"] == {"valid": True, "reasons": [], "bad_channels": []}
    assert set(payload) == set(frame.to_dict())
    assert "NaN" not in frame.to_json()


def test_decision_states_are_exactly_six() -> None:
    assert {state.value for state in DecisionState} == {
        "warming_up",
        "idle",
        "left_hand",
        "right_hand",
        "uncertain",
        "invalid",
    }
    assert set(QUALITY_REASONS) == {
        "non_finite",
        "missing_channel",
        "sample_jump",
        "interval_error",
        "flatline",
        "amplitude",
    }


# --------------------------------------------------------------------- 配置


def test_config_defaults_are_the_frozen_baseline() -> None:
    config = DEFAULT_CONFIG
    assert config.source.sample_rate_hz == 160.0
    assert len(config.source.model_channels) == 9
    assert not hasattr(config.source, "channel_count"), "契约 1.1 起通道数属于数据源，不进配置"
    assert config.source.reference_policy == "model_channel_average"
    assert config.source.reference_policy in REFERENCE_POLICIES
    assert (config.processing.window_sec, config.processing.step_sec) == (2.0, 0.25)
    assert config.processing.amplitude_limit_uv > 0
    assert config.processing.flatline_sec > 0
    assert config.processing.flatline_tolerance_uv >= 0
    assert config.decision.valid_for_ms == 750
    assert len(config.config_hash) == 16


def test_config_rejects_unknown_reference_policy() -> None:
    _expect(
        ConfigError,
        lambda: replace(DEFAULT_CONFIG.source, reference_policy="my_average"),
        "参考策略必须来自固定词表",
    )


def test_reset_reasons_are_pinned() -> None:
    assert {reason.value for reason in ResetReason} == {
        "session_start",
        "discontinuity",
        "quality_invalid",
        "pause_resume",
    }


# --------------------------------------------------------------------- 事件与会话


def test_intent_event_open_and_closed() -> None:
    open_event = IntentEvent(
        event_id="e1",
        session_id="demo",
        recording_id="S001R04",
        state=DecisionState.LEFT_HAND,
        start_sample=480,
        start_decision_seq=12,
    )
    assert open_event.is_open
    assert open_event.duration_samples is None
    assert open_event.duration_sec(SAMPLE_RATE) is None

    closed = IntentEvent(
        event_id="e2",
        session_id="demo",
        recording_id="S001R04",
        state=DecisionState.RIGHT_HAND,
        start_sample=480,
        start_decision_seq=12,
        end_sample=800,
        end_decision_seq=20,
        end_reason="state_changed",
    )
    assert closed.duration_samples == 320
    assert closed.duration_sec(SAMPLE_RATE) == 2.0
    assert closed.end_reason in END_REASONS
    assert set(closed.to_dict()) == {
        "event_id",
        "session_id",
        "recording_id",
        "state",
        "start_sample",
        "start_decision_seq",
        "end_sample",
        "end_decision_seq",
        "end_reason",
    }

    _expect(
        ContractError,
        lambda: IntentEvent(
            event_id="e3",
            session_id="demo",
            recording_id="S001R04",
            state=DecisionState.LEFT_HAND,
            start_sample=0,
            start_decision_seq=0,
            end_sample=320,
        ),
        "结束字段必须齐全",
    )
    _expect(
        ContractError,
        lambda: IntentEvent(
            event_id="e4",
            session_id="demo",
            recording_id="S001R04",
            state=DecisionState.LEFT_HAND,
            start_sample=480,
            start_decision_seq=12,
            end_sample=480,
            end_decision_seq=20,
            end_reason="state_changed",
        ),
        "end_sample 必须大于 start_sample",
    )
    _expect(
        ContractError,
        lambda: IntentEvent(
            event_id="e5",
            session_id="demo",
            recording_id="S001R04",
            state=DecisionState.IDLE,
            start_sample=0,
            start_decision_seq=0,
        ),
        "只有左右手能成为动作事件",
    )
    _expect(
        ContractError,
        lambda: IntentEvent(
            event_id="e6",
            session_id="demo",
            recording_id="S001R04",
            state=DecisionState.LEFT_HAND,
            start_sample=0,
            start_decision_seq=0,
            end_sample=320,
            end_decision_seq=10,
            end_reason="oops",
        ),
        "结束原因必须在词表内",
    )


def test_runtime_event_and_session_state() -> None:
    assert [state.value for state in SessionState] == [
        "ready",
        "running",
        "paused",
        "finished",
        "stopped",
        "error",
    ]
    event = RuntimeEvent(
        session_id="demo", at_monotonic_ms=12.5, kind=RuntimeEventKind.PAUSE, detail=" 用户暂停 "
    )
    assert event.to_dict() == {
        "session_id": "demo",
        "at_monotonic_ms": 12.5,
        "kind": "pause",
        "detail": "用户暂停",
    }
    _expect(
        ContractError,
        lambda: RuntimeEvent(session_id="demo", at_monotonic_ms=1.0, kind="pause"),
        "kind 必须是枚举",
    )


# --------------------------------------------------------------------- 模型包


def test_model_input_spec_rules() -> None:
    spec = _model_spec()
    assert spec.feature_count == 18
    assert spec.classes == ("idle", "left_hand", "right_hand")
    assert spec.feature_layout in FEATURE_LAYOUTS
    assert len(spec.weights_sha256) == 64

    _expect(
        ContractError,
        lambda: _model_spec(classes=("left_hand", "idle", "right_hand")),
        "类别顺序固定，左右手不能换",
    )
    _expect(ContractError, lambda: _model_spec(weights_sha256="abc"), "权重 SHA-256 格式")
    _expect(ContractError, lambda: _model_spec(feature_layout="row_major"), "特征布局词表")
    _expect(ContractError, lambda: _model_spec(preprocessing_hash="zz"), "预处理指纹格式")
    _expect(
        ContractError,
        lambda: _model_spec(channel_positions={"XX": (0.0, 0.0, 0.0)}),
        "坐标键必须在 channels 内",
    )


def test_model_spec_compatibility() -> None:
    spec = _model_spec()
    spec.check_compatible(_info(), DEFAULT_CONFIG)  # 应当通过

    bad_rate = _info(sample_rate_hz=250.0)
    try:
        spec.check_compatible(bad_rate, DEFAULT_CONFIG)
    except IncompatibleModelError as error:
        assert any("采样率" in reason for reason in error.reasons)
    else:
        raise AssertionError("采样率不符必须拒绝")

    other_config = replace(
        DEFAULT_CONFIG,
        processing=replace(DEFAULT_CONFIG.processing, warmup_sec=4.0),
    )
    try:
        spec.check_compatible(_info(), other_config)
    except IncompatibleModelError as error:
        assert any("预处理指纹" in reason for reason in error.reasons)
    else:
        raise AssertionError("预处理指纹不符必须拒绝")


def test_preprocessing_hash_ignores_ui_and_chunking() -> None:
    base = DEFAULT_CONFIG.preprocessing_hash
    ui_only = replace(
        DEFAULT_CONFIG, display=replace(DEFAULT_CONFIG.display, target_refresh_hz=30.0)
    )
    buffer_only = replace(
        DEFAULT_CONFIG, runtime=replace(DEFAULT_CONFIG.runtime, ring_buffer_sec=60.0)
    )
    replay_only = replace(
        DEFAULT_CONFIG,
        source=replace(DEFAULT_CONFIG.source, target_chunk_ms=100.0, replay_speed=2.0),
    )
    for other in (ui_only, buffer_only, replay_only):
        assert other.preprocessing_hash == base
    assert ui_only.config_hash != DEFAULT_CONFIG.config_hash

    signal_changed = replace(
        DEFAULT_CONFIG, processing=replace(DEFAULT_CONFIG.processing, warmup_sec=4.0)
    )
    assert signal_changed.preprocessing_hash != base


def test_source_info_hashable_and_path_sha() -> None:
    first = _info(path="data/eegmmidb/S001/S001R04.edf", sha256="c" * 64)
    second = _info(path="data/eegmmidb/S001/S001R04.edf", sha256="c" * 64)
    assert hash(first) == hash(second)
    assert first == second
    _expect(ContractError, lambda: _info(sha256="xyz"), "sha256 必须 64 位十六进制")


def test_quality_reasons_are_canonically_ordered() -> None:
    report = QualityReport.failed(("flatline", "non_finite"))
    assert report.reasons == ("non_finite", "flatline")


def _run_all() -> int:
    names = [name for name in sorted(globals()) if name.startswith("test_")]
    failures = 0
    for name in names:
        try:
            globals()[name]()
        except AssertionError as error:
            failures += 1
            print(f"FAIL {name}: {error}")
        except Exception as error:  # noqa: BLE001
            failures += 1
            print(f"ERROR {name}: {type(error).__name__}: {error}")
        else:
            print(f"ok   {name}")
    print(f"\n{len(names) - failures} passed, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_run_all())
