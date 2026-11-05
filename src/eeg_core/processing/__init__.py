"""处理链：质量与因果滤波归 A，分窗与特征归 B。

``quality.py`` / ``filters.py``（A 维护）
    ``QualityChecker`` 只检查原始块里的模型通道；``FilterBank.process`` 串联质量、模型通道平均参考与
    两路 Butterworth SOS 因果滤波，返回定位元数据、独立质量报告和两路数组，不创建重复
    ``seq`` 的 ``EEGChunk``。间断或质量异常清理本模块历史并重新预热；
    ``ResetReason.PAUSE_RESUME`` 保留滤波状态与已处理历史。

``windows.py`` / ``features.py``（B 维护）
    在**已滤波**数据上按 2 s 窗 / 250 ms 步长分窗，算每通道每频段的对数方差（18 维）。
    **不重复实现质量规则与滤波**，见 ``docs/01_接口约定_1.0.md`` §3.1、§3.2。

依赖方向：``processing`` 只依赖 ``contracts``，不反向依赖 ``eeg_app`` 层。
"""

from .features import FEATURE_DIM, log_variance_features
from .filters import FilterBank
from .quality import QualityChecker
from .windows import (
    hop_samples,
    is_window_usable,
    iter_window_ranges,
    required_warmup_samples,
    window_samples,
)

__all__ = [
    "FEATURE_DIM",
    "FilterBank",
    "QualityChecker",
    "hop_samples",
    "is_window_usable",
    "iter_window_ranges",
    "log_variance_features",
    "required_warmup_samples",
    "window_samples",
]
