"""配置契约：一次运行的全局工程参数，唯一出处（README §8 / PRD §6.1）。

下游模块（sources / runtime / processing / decoding / ui / outputs）只许从这里读参数，
不得各自另写默认值——否则同一次运行会出现「界面按 20 Hz 刷新、解码按 10 Hz 推进」这类
对不上的情况，而且没人能说清当时到底用的哪套数。

三条使用规则：

1. **只读**：所有配置都是 ``frozen`` 数据类，运行中不能改。要换参数就换一份配置重新跑。
   **结构或语义变更**（加/删字段、改单位、改时间口径）必须升 :data:`CONTRACT_VERSION` 并同步
   docs/01；**纯数值调整**（门限、有效期、刷新率等）不升版本，只通过 :attr:`AppConfig.config_hash`
   写进 manifest 与运行日志——否则验证阶段每调一次阈值就要发一次版本。
2. **可溯源**：:attr:`AppConfig.config_hash` 覆盖全部字段，写进 manifest 与
   ``DecisionFrame.config_hash``；同一份配置在任何机器上都得到同一个 hash。
3. **与模型包一致**：通道表、采样率、滤波、参考方式与门限必须与模型包记录一致，不一致时拒绝运行
   （T02），不能在运行时「凑合换一个」。

待验证数值：对数方差下限仍是占位值（见 :class:`ProcessingConfig`）；三个质量阈值已于
2026-11-05 用 S001–S010 标定（``scripts/calibrate_quality.py``）。补字段同样要升契约版本。
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
from dataclasses import asdict, dataclass, field

from .frames import CONTRACT_VERSION

__all__ = [
    "DEFAULT_CONFIG",
    "MODEL_CHANNELS",
    "REFERENCE_POLICIES",
    "AppConfig",
    "ConfigError",
    "DecisionConfig",
    "DisplayConfig",
    "ProcessingConfig",
    "RuntimeConfig",
    "SourceConfig",
]

MODEL_CHANNELS = ("FC3", "FCz", "FC4", "C3", "Cz", "C4", "CP3", "CPz", "CP4")
"""模型使用的 9 个通道；顺序即特征向量顺序，训练与运行必须一致。"""

REFERENCE_POLICIES = (
    "as_recorded",
    "model_channel_average",
    "average_all",
    "linked_mastoid",
    "cz",
)
"""``SourceConfig.reference_policy`` 的固定词表。

``as_recorded`` 表示不额外重参考、直接用文件里记录的参考（首版默认）；其余取值定义应用层
自己的重参考方式。**训练与运行时必须一致**，否则分布错位；实际取值写进模型包的预处理指纹。
"""


class ConfigError(ValueError):
    """配置不合法：类型不对、越界，或字段之间的组合关系不成立。"""


def _require_text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise ConfigError(f"{field_name} 必须是字符串，收到 {type(value).__name__}")
    text = value.strip()
    if not text:
        raise ConfigError(f"{field_name} 不能为空")
    return text


def _require_int(value: object, field_name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ConfigError(f"{field_name} 必须是整数，收到 {type(value).__name__}")
    number = int(value)
    if number < minimum:
        raise ConfigError(f"{field_name} 不能小于 {minimum}，收到 {number}")
    return number


def _require_real(
    value: object,
    field_name: str,
    *,
    minimum: float | None = None,
    strictly_positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ConfigError(f"{field_name} 必须是实数，收到 {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise ConfigError(f"{field_name} 必须是有限值，收到 {number!r}")
    if strictly_positive and number <= 0.0:
        raise ConfigError(f"{field_name} 必须大于 0，收到 {number!r}")
    if minimum is not None and number < minimum:
        raise ConfigError(f"{field_name} 不能小于 {minimum}，收到 {number!r}")
    return number


def _require_channels(values: object, field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ConfigError(f"{field_name} 必须是通道名序列，不能是单个字符串")
    try:
        items = list(values)  # type: ignore[arg-type]
    except TypeError:
        raise ConfigError(f"{field_name} 必须是通道名序列") from None
    if not items:
        raise ConfigError(f"{field_name} 不能为空")

    names: list[str] = []
    for index, item in enumerate(items):
        if not isinstance(item, str) or not item.strip():
            raise ConfigError(f"{field_name}[{index}] 必须是非空字符串")
        names.append(item.strip())
    if len(set(names)) != len(names):
        raise ConfigError(f"{field_name} 存在重复通道名")
    return tuple(names)


def _require_bands(values: object) -> tuple[tuple[float, float], ...]:
    try:
        items = list(values)  # type: ignore[arg-type]
    except TypeError:
        raise ConfigError("bands_hz 必须是 (低频, 高频) 序列") from None
    if not items:
        raise ConfigError("bands_hz 不能为空")

    bands: list[tuple[float, float]] = []
    for index, item in enumerate(items):
        try:
            low, high = item
        except (TypeError, ValueError):
            raise ConfigError(f"bands_hz[{index}] 必须是 (低频, 高频) 二元组") from None
        low_hz = _require_real(low, f"bands_hz[{index}].low", strictly_positive=True)
        high_hz = _require_real(high, f"bands_hz[{index}].high", strictly_positive=True)
        if low_hz >= high_hz:
            raise ConfigError(f"bands_hz[{index}] 低频必须小于高频，收到 ({low_hz:g}, {high_hz:g})")
        bands.append((low_hz, high_hz))
    return tuple(bands)


@dataclass(frozen=True, slots=True)
class SourceConfig:
    """**算法**对数据的要求 + 回放节奏；不描述数据源本身。

    这里只回答「算法要什么」：目标采样率、模型需要的通道、应用自己做的重参考策略、
    回放速度与分块时长。数据源有多少通道、通道叫什么、什么格式、是哪个设备，
    由 ``SourceInfo`` 在运行时自描述（契约 1.1 起不再有 ``channel_count``）。

    ``sample_rate_hz`` 是**算法目标采样率**：滤波器系数、窗长/步长样本数都按它算。
    数据源采样率不同时，由数据源层显式重采样到它并记录在 ``SourceInfo.device_metadata``，
    不能靠改配置来“凑”不同设备。

    ``reference_policy`` 是**应用自己**要做的重参考（训练与运行必须一致），
    不是文件里写的参考方式——后者在 ``SourceInfo.reference``，缺失时记为「未提供」。
    取值只能来自 :data:`REFERENCE_POLICIES`；首版用 ``model_channel_average``，
    由 A 的 ``FilterBank`` 实现，训练必须调用同一份实现。

    ``model_channels`` 是**模型要求**的通道名；记录缺任何一个都由 ``check_compatible``
    显式报错，不是 ``EEGChunk`` 的构造错误。改名/别名由数据源层的别名表处理。
    """

    sample_rate_hz: float = 160.0
    model_channels: tuple[str, ...] = MODEL_CHANNELS
    reference_policy: str = "model_channel_average"
    replay_speed: float = 1.0
    target_chunk_ms: float = 50.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "sample_rate_hz",
            _require_real(self.sample_rate_hz, "sample_rate_hz", strictly_positive=True),
        )
        model_channels = _require_channels(self.model_channels, "model_channels")
        object.__setattr__(self, "model_channels", model_channels)
        object.__setattr__(
            self, "reference_policy", _require_text(self.reference_policy, "reference_policy")
        )
        if self.reference_policy not in REFERENCE_POLICIES:
            raise ConfigError(
                f"reference_policy 必须是 {REFERENCE_POLICIES} 之一，收到 {self.reference_policy!r}"
            )
        object.__setattr__(
            self,
            "replay_speed",
            _require_real(self.replay_speed, "replay_speed", strictly_positive=True),
        )
        object.__setattr__(
            self,
            "target_chunk_ms",
            _require_real(self.target_chunk_ms, "target_chunk_ms", strictly_positive=True),
        )


@dataclass(frozen=True, slots=True)
class ProcessingConfig:
    """处理与特征：窗口、频带、滤波、预热与质量阈值。

    质量阈值单位是**微伏 / 秒**（与界面一致），由 ``app.py`` 组装时乘 ``1e-6`` 传给
    :class:`eeg_core.processing.quality.QualityChecker`（它的参数单位是伏），换算只做一次。
    三个阈值 2026-11-05 用 S001–S010 标定，取值理由见各字段的 docstring 与
    ``scripts/calibrate_quality.py``；换数据源（尤其量化步长不同）要按同一脚本重标。
    """

    window_sec: float = 2.0
    step_sec: float = 0.25
    bands_hz: tuple[tuple[float, float], ...] = ((8.0, 13.0), (13.0, 30.0))
    filter_order: int = 4
    filter_impl: str = "butterworth_sos"
    log_variance_floor: float = 1e-12
    warmup_sec: float = 3.0
    amplitude_limit_uv: float = 500.0
    """单样本绝对值上限（µV），超过判 ``amplitude``。

    2026-11-05 用 S001–S010 的 30 条记录标定（``scripts/calibrate_quality.py``）：模型通道
    每块最大 ``|x|`` 的 p99.99 约 499 µV；500 落在伪迹尾部拐点上（400 µV → 0.66% 无效块，
    500 → 0.07%，600 → 0.004%），再放宽只是少拦几段本来就不能用的数据。
    """

    flatline_sec: float = 1.0
    """连续常量段达到此时长（秒）判 ``flatline``。

    生理信号在 160 Hz 下不会连续 1 s 相邻差不变（30 条记录实测最长 0.49 s，且出现在文件
    尾补齐）；能跑满 1 s 的只有死通道/卡死通道，检测延迟与 3 s 预热相比可接受。
    """

    flatline_tolerance_uv: float = 1.0
    """相邻样本绝对差 ≤ 此值（µV）视为未变化；0 表示严格相等。

    取 EEGMMIDB 的量化步长（相邻样本最小非零差实测 1.000 µV）。容差小于一步量化时
    “未变化”会退化成“严格相等”：死通道只要在相邻两个码值间抖动（±1 LSB），就永远
    判不出常量段。换更细量化的数据源时，这里应按该源的 LSB 重新标定。
    """

    def __post_init__(self) -> None:
        window = _require_real(self.window_sec, "window_sec", strictly_positive=True)
        step = _require_real(self.step_sec, "step_sec", strictly_positive=True)
        if step > window:
            raise ConfigError(f"step_sec 不能大于 window_sec，收到 {step:g} > {window:g}")
        object.__setattr__(self, "window_sec", window)
        object.__setattr__(self, "step_sec", step)
        object.__setattr__(self, "bands_hz", _require_bands(self.bands_hz))
        object.__setattr__(
            self, "filter_order", _require_int(self.filter_order, "filter_order", minimum=1)
        )
        object.__setattr__(self, "filter_impl", _require_text(self.filter_impl, "filter_impl"))
        object.__setattr__(
            self,
            "log_variance_floor",
            _require_real(self.log_variance_floor, "log_variance_floor", strictly_positive=True),
        )
        object.__setattr__(
            self, "warmup_sec", _require_real(self.warmup_sec, "warmup_sec", minimum=0.0)
        )
        object.__setattr__(
            self,
            "amplitude_limit_uv",
            _require_real(self.amplitude_limit_uv, "amplitude_limit_uv", strictly_positive=True),
        )
        object.__setattr__(
            self,
            "flatline_sec",
            _require_real(self.flatline_sec, "flatline_sec", strictly_positive=True),
        )
        object.__setattr__(
            self,
            "flatline_tolerance_uv",
            _require_real(self.flatline_tolerance_uv, "flatline_tolerance_uv", minimum=0.0),
        )


@dataclass(frozen=True, slots=True)
class DecisionConfig:
    """连续决策：门限、确认次数、结果有效期与分类器参数。"""

    threshold: float = 0.65
    margin: float = 0.15
    confirm_updates: int = 3
    valid_for_ms: int = 750
    lda_solver: str = "lsqr"
    lda_shrinkage: str = "auto"

    def __post_init__(self) -> None:
        threshold = _require_real(self.threshold, "threshold", strictly_positive=True)
        if threshold >= 1.0:
            raise ConfigError(f"threshold 必须小于 1，收到 {threshold!r}")
        margin = _require_real(self.margin, "margin", minimum=0.0)
        if margin >= 1.0:
            raise ConfigError(f"margin 不能大于等于 1，收到 {margin!r}")
        if threshold + margin > 1.0:
            raise ConfigError(
                f"threshold + margin 不能超过 1，收到 {threshold:g} + {margin:g}"
            )
        object.__setattr__(self, "threshold", threshold)
        object.__setattr__(self, "margin", margin)
        object.__setattr__(
            self,
            "confirm_updates",
            _require_int(self.confirm_updates, "confirm_updates", minimum=1),
        )
        object.__setattr__(
            self, "valid_for_ms", _require_int(self.valid_for_ms, "valid_for_ms", minimum=1)
        )
        object.__setattr__(self, "lda_solver", _require_text(self.lda_solver, "lda_solver"))
        object.__setattr__(
            self, "lda_shrinkage", _require_text(self.lda_shrinkage, "lda_shrinkage")
        )


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """运行时会话：环形缓冲与积压（过载、暂停）阈值。"""

    ring_buffer_sec: float = 30.0
    overload_warn_ms: float = 500.0
    overrun_pause_ms: float = 1000.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "ring_buffer_sec",
            _require_real(self.ring_buffer_sec, "ring_buffer_sec", strictly_positive=True),
        )
        warn = _require_real(self.overload_warn_ms, "overload_warn_ms", strictly_positive=True)
        pause = _require_real(self.overrun_pause_ms, "overrun_pause_ms", strictly_positive=True)
        if pause <= warn:
            raise ConfigError(
                f"overrun_pause_ms 必须大于 overload_warn_ms，收到 {pause:g} <= {warn:g}"
            )
        object.__setattr__(self, "overload_warn_ms", warn)
        object.__setattr__(self, "overrun_pause_ms", pause)


@dataclass(frozen=True, slots=True)
class DisplayConfig:
    """界面：波形通道数、时间跨度与刷新率目标。"""

    waveform_channels: int = 8
    waveform_span_sec: float = 10.0
    target_refresh_hz: float = 20.0
    min_refresh_hz: float = 15.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "waveform_channels",
            _require_int(self.waveform_channels, "waveform_channels", minimum=1),
        )
        object.__setattr__(
            self,
            "waveform_span_sec",
            _require_real(self.waveform_span_sec, "waveform_span_sec", strictly_positive=True),
        )
        target = _require_real(self.target_refresh_hz, "target_refresh_hz", strictly_positive=True)
        minimum = _require_real(self.min_refresh_hz, "min_refresh_hz", strictly_positive=True)
        if minimum > target:
            raise ConfigError(
                f"min_refresh_hz 不能大于 target_refresh_hz，收到 {minimum:g} > {target:g}"
            )
        object.__setattr__(self, "target_refresh_hz", target)
        object.__setattr__(self, "min_refresh_hz", minimum)


@dataclass(frozen=True, slots=True)
class AppConfig:
    """一次运行的完整配置；``DEFAULT_CONFIG`` 是唯一默认实例。"""

    source: SourceConfig = field(default_factory=SourceConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    contract_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        version = _require_text(self.contract_version, "contract_version")
        if version != CONTRACT_VERSION:
            raise ConfigError(
                f"不支持的 contract_version {version!r}，当前契约版本为 {CONTRACT_VERSION}"
            )
        object.__setattr__(self, "contract_version", version)

    def to_dict(self) -> dict[str, object]:
        """可 JSON 化的普通字典；元组按数组写出，字段顺序按分组。"""
        data = asdict(self)
        data["source"]["model_channels"] = list(self.source.model_channels)
        data["processing"]["bands_hz"] = [list(band) for band in self.processing.bands_hz]
        return data

    @property
    def config_hash(self) -> str:
        """全部字段的规范指纹，用于 manifest 与 ``DecisionFrame.config_hash``。"""
        payload = json.dumps(
            self.to_dict(), sort_keys=True, ensure_ascii=False, separators=(",", ":")
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    @property
    def preprocessing_hash(self) -> str:
        """**只覆盖信号路径**的指纹：采样率、模型通道、参考、滤波、窗、频带。

        与 :attr:`config_hash` 的区别：界面刷新率、环形缓冲、回放速度、分块大小、
        门限有效期等变化**不会**改变它。T02 用它比对模型包，避免“改个 UI 参数就
        报模型不兼容”，也避免“改个块长就要求重训”。``DecisionConfig`` 不参与，
        因为门限允许按受试者在验证阶段调整，实际取值由模型包记录。

        数据源通道数不在这里：记录有多少通道由 ``SourceInfo`` 自描述，换 22 导设备
        不会改变算法指纹。
        """
        payload = {
            "source": {
                "sample_rate_hz": self.source.sample_rate_hz,
                "model_channels": list(self.source.model_channels),
                "reference_policy": self.source.reference_policy,
            },
            "processing": {
                "window_sec": self.processing.window_sec,
                "step_sec": self.processing.step_sec,
                "bands_hz": [list(band) for band in self.processing.bands_hz],
                "filter_order": self.processing.filter_order,
                "filter_impl": self.processing.filter_impl,
                "log_variance_floor": self.processing.log_variance_floor,
                "warmup_sec": self.processing.warmup_sec,
                "amplitude_limit_uv": self.processing.amplitude_limit_uv,
                "flatline_sec": self.processing.flatline_sec,
                "flatline_tolerance_uv": self.processing.flatline_tolerance_uv,
            },
        }
        text = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


DEFAULT_CONFIG = AppConfig()
"""初始工程配置；契约 1.1——改结构/语义须升 ``CONTRACT_VERSION``，改数值只记 ``config_hash``。"""
