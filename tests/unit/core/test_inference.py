"""模型包加载：model.json + weights.npz 的校验与推理。"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from eeg_core.contracts import (
    MODEL_CHANNELS,
    ContractError,
    DecisionConfig,
    ModelInputSpec,
)
from eeg_core.decoding import (
    CLASSES,
    fit_lda,
    load_model_package,
    save_weights,
)
from eeg_core.version import PIPELINE_VERSION

BASE = Path("training/models")


def _dataset() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20261103)
    features = rng.normal(size=(90, 18))
    labels = np.asarray([CLASSES[index % 3] for index in range(90)])
    return features, labels


def _spec(
    model_id: str,
    weights_sha: str,
    *,
    pipeline_version: str = PIPELINE_VERSION,
) -> ModelInputSpec:
    return ModelInputSpec(
        model_id=model_id,
        pipeline_version=pipeline_version,
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
        weights_sha256=weights_sha,
        preprocessing_hash="0" * 16,
    )


def _write_package(
    *,
    name: str,
    pipeline_version: str = PIPELINE_VERSION,
) -> Path:
    features, labels = _dataset()
    arrays, _ = fit_lda(features, labels)
    weights = BASE / f"{name}.weights.npz"
    actual_sha = save_weights(weights, arrays)
    spec = _spec(name, actual_sha, pipeline_version=pipeline_version)
    payload = {
        "spec": spec.to_dict(),
        "decision": asdict(spec.decision),
        "weights_file": weights.name,
    }
    model_json = BASE / f"{name}.model.json"
    model_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return model_json


def test_roundtrip_load_and_scores() -> None:
    name = ".unit-test-inference-ok"
    model_json = _write_package(name=name)
    weights = BASE / f"{name}.weights.npz"
    try:
        package = load_model_package(model_json)
        assert package.model_id == name
        assert package.spec.pipeline_version == PIPELINE_VERSION
        features, _ = _dataset()
        scores = package.scores(features[0])
        assert abs(scores.idle + scores.left_hand + scores.right_hand - 1.0) < 1e-9
        assert package.predict_label(features[0]) in CLASSES
    finally:
        model_json.unlink(missing_ok=True)
        weights.unlink(missing_ok=True)


def test_wrong_pipeline_version_is_rejected() -> None:
    name = ".unit-test-inference-old"
    model_json = _write_package(name=name, pipeline_version="old-pipe")
    weights = BASE / f"{name}.weights.npz"
    try:
        with pytest.raises(ContractError, match="pipeline_version"):
            load_model_package(model_json)
    finally:
        model_json.unlink(missing_ok=True)
        weights.unlink(missing_ok=True)


def test_tampered_weights_are_rejected() -> None:
    name = ".unit-test-inference-sha"
    model_json = _write_package(name=name)
    weights = BASE / f"{name}.weights.npz"
    try:
        payload = json.loads(model_json.read_text(encoding="utf-8"))
        payload["spec"]["weights_sha256"] = "f" * 64
        model_json.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ContractError, match="SHA-256"):
            load_model_package(model_json)
    finally:
        model_json.unlink(missing_ok=True)
        weights.unlink(missing_ok=True)
