"""封存测试集 run12：加载层默认拒绝，必须显式放行。

这是「持续封存」的代码闸门，不是文档约定：任何脚本（train / evaluate / calibrate）
只要走 ``load_edf`` / ``load_recording``，默认就读不到 run12。

测试刻意用不存在的路径：封存判断在**读文件之前**，所以不需要真实 EDF，也不写磁盘。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from eeg_app.sources.input import load_edf, load_recording
from eeg_core.contracts import ContractError

MISSING = Path("_no_such_dir") / "S001R12.edf"


def test_load_edf_rejects_sealed_run_by_default() -> None:
    with pytest.raises(ContractError, match="封存测试集"):
        load_edf(MISSING)


def test_load_edf_allows_sealed_run_only_with_flag() -> None:
    # 过了封存闸门，才轮到“文件不存在”；说明 allow_sealed=True 确实放行。
    with pytest.raises(ContractError, match="找不到 EDF 文件"):
        load_edf(MISSING, allow_sealed=True)


def test_open_run_is_not_sealed() -> None:
    with pytest.raises(ContractError, match="找不到 EDF 文件"):
        load_edf(MISSING.with_name("S001R04.edf"))


def test_load_recording_rejects_sealed_run() -> None:
    with pytest.raises(ContractError, match="封存测试集"):
        load_recording(MISSING.parent, "S001R12")


def test_load_recording_allows_sealed_run_only_with_flag() -> None:
    with pytest.raises(ContractError, match="找不到记录文件"):
        load_recording(MISSING.parent, "S001R12", allow_sealed=True)
