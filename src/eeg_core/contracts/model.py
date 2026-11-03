"""契约 1.0 的模型包：训练产物的“身份证”，T02 用它做启动校验。

模型包 = 权重文件 + 一份 :class:`ModelInputSpec`。本模块只定义**规格**，不负责读写
``.npz`` 或反序列化分类器——那是 ``decoding/`` 的实现，按纪律不提前抽象。

四条硬规则：

1. **规格即契约**：通道顺序、参考方式、频带、窗长/步长、特征布局、类别顺序、门限、
   权重与训练文件的 SHA-256 全部写在规格里；缺一项就拒绝加载。
2. **特征是“通道 × 频带”的顺序**：``feature_layout`` 固定为 ``channel_band``（外层通道、
   内层频带），训练与推理必须同序，错一位就是静默错误。
3. **预处理指纹**：``preprocessing_hash`` 由 :attr:`AppConfig.preprocessing_hash` 生成，
   只覆盖采样率、通道、参考、滤波、窗、频带与决策参数；界面/缓冲/回放速度变化不影响它。
4. **不猜**：对不上就抛 :class:`IncompatibleModelError` 并一次列全原因（T02），
   不做“自动凑合”。
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .config import REFERENCE_POLICIES, DecisionConfig
from .frames import (
    ContractError,
    SourceInfo,
    _require_channel_names,
    _require_real,
    _require_text,
)

__all__ = [
    "FEATURE_LAYOUTS",
    "IncompatibleModelError",
    "ModelInputSpec",
]

FEATURE_LAYOUTS = ("channel_band", "band_channel")
"""特征向量布局；首版固定 ``channel_band``（先 9 个通道、每个通道内 2 个频带，共 18 维）。"""

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_HASH16 = re.compile(r"^[0-9a-f]{16}$")


class IncompatibleModelError(ContractError):
    """模型包与记录或运行配置不兼容，一次性携带全部原因（T02 的“模型侧”）。"""

    def __init__(self, model_id: str, reasons: tuple[str, ...] | list[str]) -> None:
        self.model_id = model_id
        self.reasons = tuple(reasons)
        if not self.reasons:
            raise ValueError("IncompatibleModelError 至少需要一个原因")
        super().__init__(f"模型 {model_id} 不适用：" + "；".join(self.reasons))


def _require_sha256(value: object, field_name: str) -> str:
    text = _require_text(value, field_name).lower()
    if _SHA256.match(text) is None:
        raise ContractError(f"{field_name} 必须是 64 位十六进制 SHA-256，收到 {value!r}")
    return text


def _require_hash16(value: object, field_name: str) -> str:
    text = _require_text(value, field_name).lower()
    if _HASH16.match(text) is None:
        raise ContractError(f"{field_name} 必须是 16 位十六进制指纹，收到 {value!r}")
    return text


def _require_bands(values: object) -> tuple[tuple[float, float], ...]:
    try:
        items = list(values)  # type: ignore[arg-type]
    except TypeError:
        raise ContractError("bands_hz 必须是 (低频, 高频) 序列") from None
    if not items:
        raise ContractError("bands_hz 不能为空")

    bands: list[tuple[float, float]] = []
    for index, item in enumerate(items):
        try:
            low, high = item
        except (TypeError, ValueError):
            raise ContractError(f"bands_hz[{index}] 必须是二元组") from None
        low_hz = _require_real(low, f"bands_hz[{index}].low", strictly_positive=True)
        high_hz = _require_real(high, f"bands_hz[{index}].high", strictly_positive=True)
        if low_hz >= high_hz:
            raise ContractError(
                f"bands_hz[{index}] 低频必须小于高频，收到 ({low_hz:g}, {high_hz:g})"
            )
        bands.append((low_hz, high_hz))
    return tuple(bands)


def _optional_positions(value: object, channels: tuple[str, ...]):
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ContractError("channel_positions 必须是 通道名 → (x, y, z) 的映射")
    known = set(channels)
    positions: dict[str, tuple[float, float, float]] = {}
    for name, xyz in value.items():
        if not isinstance(name, str) or name not in known:
            raise ContractError(f"channel_positions 里的通道 {name!r} 不在 channels 中")
        try:
            x, y, z = xyz  # type: ignore[misc]
        except (TypeError, ValueError):
            raise ContractError(f"channel_positions[{name}] 必须是三个数字") from None
        positions[name] = (
            _require_real(x, f"channel_positions[{name}].x"),
            _require_real(y, f"channel_positions[{name}].y"),
            _require_real(z, f"channel_positions[{name}].z"),
        )
    return MappingProxyType(positions)


def _optional_files(value: object) -> Mapping[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ContractError("training_files 必须是 路径 → SHA-256 的映射")
    files: dict[str, str] = {}
    for path, digest in value.items():
        key = _require_text(path, "training_files 的键")
        files[key] = _require_sha256(digest, f"training_files[{key}]")
    return MappingProxyType(files)


@dataclass(frozen=True, slots=True)
class ModelInputSpec:
    """一个模型包对输入的要求，训练产出、启动校验、评估复现都以它为准。

    类别顺序固定为 ``("idle", "left_hand", "right_hand")``，拒绝其它顺序：
    左右手互换是最典型的“看着正常但全反了”的失败。
    """

    model_id: str
    pipeline_version: str
    channels: tuple[str, ...]
    sample_rate_hz: float
    reference_policy: str
    bands_hz: tuple[tuple[float, float], ...]
    window_sec: float
    step_sec: float
    feature_layout: str
    feature_floor: float
    classes: tuple[str, ...]
    decision: DecisionConfig
    weights_sha256: str
    preprocessing_hash: str
    weights_format: str = "lda_npz_v1"
    channel_positions: Mapping[str, tuple[float, float, float]] | None = None
    training_files: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", _require_text(self.model_id, "model_id"))
        object.__setattr__(
            self, "pipeline_version", _require_text(self.pipeline_version, "pipeline_version")
        )
        channels = _require_channel_names(self.channels)
        object.__setattr__(self, "channels", channels)
        object.__setattr__(
            self,
            "sample_rate_hz",
            _require_real(self.sample_rate_hz, "sample_rate_hz", strictly_positive=True),
        )
        policy = _require_text(self.reference_policy, "reference_policy")
        if policy not in REFERENCE_POLICIES:
            raise ContractError(
                f"reference_policy 必须是 {REFERENCE_POLICIES} 之一，收到 {policy!r}"
            )
        object.__setattr__(self, "reference_policy", policy)
        object.__setattr__(self, "bands_hz", _require_bands(self.bands_hz))

        window = _require_real(self.window_sec, "window_sec", strictly_positive=True)
        step = _require_real(self.step_sec, "step_sec", strictly_positive=True)
        if step > window:
            raise ContractError(f"step_sec 不能大于 window_sec，收到 {step:g} > {window:g}")
        object.__setattr__(self, "window_sec", window)
        object.__setattr__(self, "step_sec", step)

        layout = _require_text(self.feature_layout, "feature_layout")
        if layout not in FEATURE_LAYOUTS:
            raise ContractError(f"feature_layout 必须是 {FEATURE_LAYOUTS} 之一，收到 {layout!r}")
        object.__setattr__(self, "feature_layout", layout)
        object.__setattr__(
            self,
            "feature_floor",
            _require_real(self.feature_floor, "feature_floor", strictly_positive=True),
        )

        classes = tuple(_require_text(name, "classes") for name in self.classes)
        if classes != ("idle", "left_hand", "right_hand"):
            raise ContractError(
                "classes 顺序必须固定为 ('idle', 'left_hand', 'right_hand')，"
                f"收到 {classes!r}"
            )
        object.__setattr__(self, "classes", classes)

        if not isinstance(self.decision, DecisionConfig):
            raise ContractError("decision 必须是 DecisionConfig")
        object.__setattr__(
            self, "weights_sha256", _require_sha256(self.weights_sha256, "weights_sha256")
        )
        object.__setattr__(
            self, "weights_format", _require_text(self.weights_format, "weights_format")
        )
        object.__setattr__(
            self,
            "preprocessing_hash",
            _require_hash16(self.preprocessing_hash, "preprocessing_hash"),
        )
        object.__setattr__(
            self, "channel_positions", _optional_positions(self.channel_positions, channels)
        )
        object.__setattr__(self, "training_files", _optional_files(self.training_files))

    @property
    def feature_count(self) -> int:
        """特征维数：通道数 × 频带数（首版 9 × 2 = 18）。"""
        return len(self.channels) * len(self.bands_hz)

    def check_compatible(self, info: SourceInfo, config) -> None:
        """核对“这个模型能不能跑这条记录/这份配置”，不通过时一次列全原因（T02）。

        ``config`` 传 :class:`eeg_core.contracts.AppConfig`；只比较影响信号的字段，
        ``preprocessing_hash`` 已把采样率、通道、参考、滤波、窗、频带与决策参数全部覆盖。
        """
        problems: list[str] = []

        missing = [name for name in self.channels if name not in info.channel_names]
        if missing:
            problems.append("记录缺少模型通道：" + "、".join(missing))
        if abs(info.sample_rate_hz - self.sample_rate_hz) > 1e-3:
            problems.append(
                f"采样率不符：模型 {self.sample_rate_hz:g} Hz，记录 {info.sample_rate_hz:g} Hz"
            )
        if tuple(config.source.model_channels) != self.channels:
            problems.append("模型通道顺序与运行配置不一致")
        if config.source.reference_policy != self.reference_policy:
            problems.append(
                f"参考策略不符：模型 {self.reference_policy!r}，"
                f"运行配置 {config.source.reference_policy!r}"
            )
        if config.preprocessing_hash != self.preprocessing_hash:
            problems.append(
                "预处理指纹不符：模型 "
                f"{self.preprocessing_hash}，运行配置 {config.preprocessing_hash}"
            )

        if problems:
            raise IncompatibleModelError(self.model_id, problems)

    def to_dict(self) -> dict[str, object]:
        """可直接写进 ``manifest.json`` 的普通字典（不含 ``decision`` 的嵌套对象）。"""
        return {
            "model_id": self.model_id,
            "pipeline_version": self.pipeline_version,
            "channels": list(self.channels),
            "sample_rate_hz": self.sample_rate_hz,
            "reference_policy": self.reference_policy,
            "bands_hz": [list(band) for band in self.bands_hz],
            "window_sec": self.window_sec,
            "step_sec": self.step_sec,
            "feature_layout": self.feature_layout,
            "feature_floor": self.feature_floor,
            "feature_count": self.feature_count,
            "classes": list(self.classes),
            "weights_sha256": self.weights_sha256,
            "weights_format": self.weights_format,
            "preprocessing_hash": self.preprocessing_hash,
            "channel_positions": (
                {name: list(xyz) for name, xyz in self.channel_positions.items()}
                if self.channel_positions is not None
                else None
            ),
            "training_files": (
                dict(self.training_files) if self.training_files is not None else None
            ),
        }
