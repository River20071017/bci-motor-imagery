"""处理链：质量与因果滤波归 A，分窗与特征归 B。

``QualityChecker`` 检查原始块；``FilterBank.process`` 串联质量、模型通道
平均参考与两路 Butterworth SOS 因果滤波，返回定位元数据、独立质量报告和
两路数组，不创建重复 seq 的 EEGChunk。间断或质量异常清理本模块历史并
重新预热；ResetReason.PAUSE_RESUME 保留滤波状态与已处理历史。

分窗与特征在已滤波数据上工作，不重复质量与滤波规则。
依赖方向：processing 只依赖 contracts，不反向依赖 eeg_app。
"""

from .filters import FilterBank
from .quality import QualityChecker

__all__ = ["FilterBank", "QualityChecker"]
