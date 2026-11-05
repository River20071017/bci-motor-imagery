"""分层约束：底层不依赖应用层，应用层不依赖训练脚本。

这不是风格检查，是防止 ``eeg_core`` 偷偷 import mne/文件/线程，最后没法单独测试、
也没法在无 IO 的环境里跑。规则见 ``docs/00_架构总览.md``。
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CORE = ROOT / "src" / "eeg_core"
APP = ROOT / "src" / "eeg_app"
TRAINING = ROOT / "training"


def _imports(path: Path) -> list[str]:
    """模块里所有 import 的目标（包括函数内 import）。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def _top_level_imports(path: Path) -> list[str]:
    """只取模块顶层 import；函数内 import 视为组合根，不在此列。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def test_core_does_not_import_app_training_or_scripts() -> None:
    for path in CORE.rglob("*.py"):
        for name in _imports(path):
            assert not name.startswith(("eeg_app", "training", "scripts")), (
                f"{path.relative_to(ROOT)} 不能 import {name}；底层必须保持无 IO/无编排"
            )


def test_app_does_not_import_training_or_scripts() -> None:
    for path in APP.rglob("*.py"):
        for name in _imports(path):
            assert not name.startswith(("training", "scripts")), (
                f"{path.relative_to(ROOT)} 不能 import {name}；应用层不能反向依赖训练脚本"
            )


def test_training_pipeline_does_not_import_app() -> None:
    """纯训练管线（pipeline/mock）不能依赖 EDF/Recording 读取。"""
    for filename in ("pipeline.py", "mock.py"):
        for name in _imports(TRAINING / filename):
            assert not name.startswith("eeg_app"), (
                f"training/{filename} 只能吃 RecordingLike，不能 import eeg_app；"
                "把文件变成 Recording 是入口 main 的职责"
            )


def test_training_entry_points_keep_app_import_out_of_module_scope() -> None:
    """入口可以依赖 app，但只能在 main/组合函数里，不能在模块顶层。"""
    for filename in ("train.py", "evaluate.py"):
        for name in _top_level_imports(TRAINING / filename):
            assert not name.startswith("eeg_app"), (
                f"training/{filename} 顶层不能 import {name}；"
                "EDF 读取应留在 main() 的组合根里"
            )
