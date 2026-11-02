"""处理链：质量与滤波归 A，分窗与特征归 B。

``quality.py`` / ``filters.py``（A 维护）
    质量判定与因果滤波：四阶 Butterworth SOS，块与块之间保留滤波器状态，
    遇到 :class:`eeg_core.contracts.ResetReason` 指定的间断、暂停或异常时清空状态并重新预热。
    产出：已滤波的 ``EEGChunk`` + 独立的 ``QualityReport``（不塞进块字段）。

``windows.py`` / ``features.py``（B 维护）
    在**已滤波**数据上按 2 s 窗 / 250 ms 步长分窗，算每通道每频段的对数方差（18 维）。
    **不重复实现滤波与质量规则**，见 ``docs/01_接口约定_1.0.md`` §3.1。

依赖方向：``processing`` 只依赖 ``contracts``，不反向依赖 ``eeg_app`` 层。
"""
