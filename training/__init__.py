"""离线训练组合层。

``pipeline.py`` 是纯管线（只吃 ``RecordingLike``，不 import ``eeg_app``）；
``train.py`` / ``evaluate.py`` 是入口，只有 ``main()`` 里接触 EDF 读取。
"""
