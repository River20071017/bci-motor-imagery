"""窗口级指标：宏 F1、左右手平衡召回、混淆矩阵。"""

from __future__ import annotations

import pytest

from eeg_core.contracts import ContractError
from training.evaluate import window_metrics


def test_perfect_predictions() -> None:
    labels = ["idle", "left_hand", "right_hand", "idle"]
    metrics = window_metrics(labels, labels)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["left_right_balanced_recall"] == 1.0
    assert metrics["confusion"]["idle"]["idle"] == 2


def test_all_idle_predicted_as_left() -> None:
    metrics = window_metrics(
        ["idle", "idle", "idle"], ["left_hand", "left_hand", "left_hand"]
    )
    assert metrics["macro_f1"] == 0.0
    assert metrics["left_right_balanced_recall"] == 0.0
    assert metrics["confusion"]["idle"]["left_hand"] == 3


def test_empty_input_rejected() -> None:
    with pytest.raises(ContractError):
        window_metrics([], [])
