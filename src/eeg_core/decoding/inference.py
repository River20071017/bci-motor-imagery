"""加载模型包（``model.json`` + ``weights.npz``），把 18 维特征变成 ``Scores``。

加载时做四道校验，缺一不可：

1. ``weights_sha256`` 与实际文件一致；
2. ``weights_format`` 是 ``lda_npz_v1``；
3. ``pipeline_version`` 与当前代码一致（防止特征/滤波代码变了但配置哈希没变）；
4. 特征维数 = ``len(channels) × len(bands_hz)``。

``ModelInputSpec.check_compatible`` 仍由运行入口在拿到 ``SourceInfo`` 后调用；
本模块只负责“模型包自身是否可用”。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from eeg_core.contracts import (
    ContractError,
    DecisionConfig,
    ModelInputSpec,
    Scores,
)
from eeg_core.decoding.models import (
    CLASSES,
    WEIGHTS_FORMAT,
    LdaArrays,
    load_weights,
    predict_proba,
    sha256_file,
)
from eeg_core.version import PIPELINE_VERSION

__all__ = ["ModelPackage", "load_model_package"]


@dataclass(frozen=True, slots=True)
class ModelPackage:
    """一个可直接推理的模型包：规格 + 数组 + 权重路径。"""

    spec: ModelInputSpec
    arrays: LdaArrays
    weights_path: Path

    @property
    def model_id(self) -> str:
        return self.spec.model_id

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """18 维特征 → 三类概率。"""
        return predict_proba(self.arrays, features)

    def scores(self, features: np.ndarray) -> Scores:
        """单个 18 维特征 → 契约 ``Scores``。"""
        probabilities = self.predict_proba(features)
        if probabilities.ndim != 1:
            raise ContractError("scores() 只接受单个 18 维特征向量")
        return Scores(
            idle=float(probabilities[0]),
            left_hand=float(probabilities[1]),
            right_hand=float(probabilities[2]),
        )

    def predict_label(self, features: np.ndarray) -> str:
        """单个 18 维特征 → 类别名（固定顺序里概率最大的那个）。"""
        probabilities = self.predict_proba(features)
        if probabilities.ndim != 1:
            raise ContractError("predict_label() 只接受单个 18 维特征向量")
        return CLASSES[int(np.argmax(probabilities))]


def _read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise ContractError(f"找不到模型描述文件：{path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ContractError(f"模型描述不是合法 JSON：{path}：{exc}") from exc
    if not isinstance(data, dict):
        raise ContractError("模型描述顶层必须是 JSON 对象")
    return data


def load_model_package(path: str | Path) -> ModelPackage:
    """读 ``model.json`` 并校验权重；失败抛 ``ContractError``，不猜不修。"""
    model_path = Path(path)
    data = _read_json(model_path)

    spec_data = data.get("spec")
    decision_data = data.get("decision")
    if not isinstance(spec_data, dict):
        raise ContractError("模型描述缺少 spec 对象")
    if not isinstance(decision_data, dict):
        raise ContractError("模型描述缺少 decision 对象")

    spec_fields = dict(spec_data)
    feature_count = spec_fields.pop("feature_count", None)
    try:
        decision = DecisionConfig(**decision_data)
        spec = ModelInputSpec(**spec_fields, decision=decision)
    except TypeError as exc:
        raise ContractError(f"模型描述字段不完整或多余：{exc}") from exc

    expected_features = len(spec.channels) * len(spec.bands_hz)
    if feature_count is not None and int(feature_count) != expected_features:
        raise ContractError(
            f"model.json 的 feature_count={feature_count} 与通道/频带推算的"
            f" {expected_features} 不一致"
        )
    if spec.feature_layout != "channel_band":
        raise ContractError(
            f"首版只支持 channel_band 布局，模型写的是 {spec.feature_layout!r}"
        )
    if spec.weights_format != WEIGHTS_FORMAT:
        raise ContractError(
            f"不支持的权重格式 {spec.weights_format!r}，当前只支持 {WEIGHTS_FORMAT}"
        )
    if spec.pipeline_version != PIPELINE_VERSION:
        raise ContractError(
            "模型 pipeline_version 与当前代码不一致："
            f"模型 {spec.pipeline_version!r}，代码 {PIPELINE_VERSION!r}；"
            "特征/滤波代码已变，请重新训练，不能静默加载"
        )

    weights_name = data.get("weights_file")
    if weights_name is None:
        weights_name = f"{spec.model_id}.weights.npz"
    if not isinstance(weights_name, str) or not weights_name.strip():
        raise ContractError("weights_file 必须是非空字符串")
    weights_path = Path(weights_name)
    if not weights_path.is_absolute():
        weights_path = model_path.parent / weights_path
    if not weights_path.is_file():
        raise ContractError(f"找不到权重文件：{weights_path}")

    actual_sha = sha256_file(weights_path)
    if actual_sha != spec.weights_sha256:
        raise ContractError(
            "权重文件 SHA-256 与模型描述不一致："
            f"文件 {actual_sha}，描述 {spec.weights_sha256}"
        )

    arrays = load_weights(weights_path)
    if arrays.n_features != expected_features:
        raise ContractError(
            f"权重特征维数 {arrays.n_features} 与模型规格 {expected_features} 不一致"
        )
    return ModelPackage(spec=spec, arrays=arrays, weights_path=weights_path)
