"""离屏冒烟测试：验证 C 的 UI 在团队契约下可运行（无需显示器）。

运行方式（在仓库根目录，使用已安装 PySide6 的虚拟环境）：

    python examples/smoke_ui_offscreen.py

覆盖要点：
1. 所有 UI 模块与契约可 import（捕获契约字段/校验错误）。
2. 主窗口可构造、可启动回放。
3. 数据块（EEGChunk）持续产生，决策（DecisionFrame）正常发布。
4. N08：开启"决策中断"后，数据继续、决策停止；超过 750 ms 无新决策，火柴人动作反馈自动静止。
5. 关闭"决策中断"后，决策恢复发布。
6. 停止后动作反馈关闭（N07）。

退出码 0 表示通过，1 表示失败；失败时打印诊断信息。
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_SRC = os.path.join(_ROOT, "src")
for _p in (_SRC, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer

import numpy as np

from eeg_core.contracts import DecisionState, EEGChunk, SourceInfo
from eeg_app.ui.main_window import MainWindow
from eeg_app.ui.intake import check_source_contract
from mock_source import MockEegSource, DEFAULT_CHANNELS
from mock_decoder import MockDecoder


def contract_coverage() -> list[str]:
    """同步校验：解码器在团队契约下能为每个状态产出合法 DecisionFrame。"""
    dec = MockDecoder(sample_rate_hz=160.0)
    dec.set_session("cov-test")
    sr = 160.0
    chs = tuple(DEFAULT_CHANNELS)
    problems: list[str] = []
    total = sum(d for d, _ in dec._script) + 5.0
    t = 0.0
    while t <= total:
        start = int(t * sr)
        chunk = EEGChunk(
            source_id="MOCK",
            recording_id="S001R12",
            seq=start // 8,
            start_sample=start,
            sample_rate_hz=sr,
            channel_names=chs,
            samples=np.zeros((len(chs), 8), dtype=np.float64),
            released_monotonic_ms=0.0,
            discontinuity=False,
        )
        try:
            df = dec.decode(chunk)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"decode@{t:.2f}s 抛异常: {exc!r}")
            t += 0.25
            continue
        if df is not None:
            if df.decision_state.is_action:
                if df.intent != df.decision_state:
                    problems.append(f"{df.decision_state} 的 intent 必须 == state")
                if df.scores is None:
                    problems.append(f"{df.decision_state} 必须有分数")
            if df.decision_state in (DecisionState.IDLE, DecisionState.UNCERTAIN):
                if df.scores is None:
                    problems.append(f"{df.decision_state} 必须有分数")
            if df.decision_state is DecisionState.WARMING_UP:
                if df.scores is not None:
                    problems.append("warming_up 不应带分数")
            if df.decision_state is DecisionState.INVALID:
                if df.quality.valid:
                    problems.append("invalid 的 quality.valid 必须为 False")
                if df.scores is not None:
                    problems.append("invalid 不应带分数")
        t += 0.25
    return problems


def main() -> int:
    app = QApplication(sys.argv)

    # 同步契约覆盖：确保每个状态都能通过团队 DecisionFrame 校验
    cov_problems = contract_coverage()
    if cov_problems:
        print("==== 契约覆盖检查 ====")
        for p in cov_problems:
            print(f"  - {p}")

    source = MockEegSource()
    decoder = MockDecoder(sample_rate_hz=source.info.sample_rate_hz)
    win = MainWindow(source=source, decoder=decoder, project_dir=_ROOT)
    win.hide()

    stats = {
        "chunks": 0,
        "decisions": 0,
        "states": [],
        "suppress_chunks": 0,
        "suppress_decisions": 0,
        "post_suppress_decisions": 0,
        "auto_stale_ok": False,
        "errors": [],
        "phase": "normal",
    }

    def on_chunk(_df):
        stats["chunks"] += 1
        if stats["phase"] == "suppress":
            stats["suppress_chunks"] += 1

    def on_decision(df):
        stats["decisions"] += 1
        stats["states"].append(df.decision_state)
        if stats["phase"] == "suppress":
            stats["suppress_decisions"] += 1
        elif stats["phase"] == "post_suppress":
            stats["post_suppress_decisions"] += 1

    # 启动回放
    win._on_start()
    worker = win.worker
    worker.chunk_ready.connect(on_chunk)
    worker.decision_ready.connect(on_decision)

    def enter_suppress():
        # N08：中断决策发布，数据继续
        stats["phase"] = "suppress"
        worker.set_suppress_decisions(True)

    def check_auto_stale_and_resume():
        # 中断已持续 > 750 ms：最后一次有效决策后动画应自动静止
        stats["auto_stale_ok"] = (win.stickman._feedback is False)
        stats["phase"] = "post_suppress"
        worker.set_suppress_decisions(False)

    def do_stop():
        win._on_stop()

    def finish():
        app.quit()

    QTimer.singleShot(600, enter_suppress)        # 运行 0.6s 后中断决策
    QTimer.singleShot(1500, check_auto_stale_and_resume)  # 0.9s 后（>750ms）检查自动静止并恢复
    QTimer.singleShot(1800, do_stop)              # 停止
    QTimer.singleShot(2100, finish)               # 收尾退出

    app.exec()

    # ---- 断言 ----
    problems = []
    if stats["chunks"] <= 0:
        problems.append("未收到任何数据块（EEGChunk）")
    if stats["decisions"] <= 0:
        problems.append("未收到任何决策（DecisionFrame）")
    if stats["suppress_decisions"] != 0:
        problems.append(f"N08 失败：中断期间仍发布 {stats['suppress_decisions']} 条决策")
    if stats["suppress_chunks"] <= 0:
        problems.append("N08 失败：中断期间数据未继续（波形应继续滚动）")
    if stats["post_suppress_decisions"] <= 0:
        problems.append("N08 失败：恢复后决策未重新发布")
    if not stats["auto_stale_ok"]:
        problems.append("N08 失败：超过 750 ms 无新决策，火柴人动作反馈未自动静止")
    if win.stickman._feedback is not False:
        problems.append("N07 失败：停止后动作反馈未关闭")
    bad_states = [s for s in stats["states"] if not isinstance(s, DecisionState)]
    if bad_states:
        problems.append(f"决策状态类型异常：{bad_states[:3]}")

    print("==== 离屏冒烟测试结果 ====")
    print(f"  数据块总数        : {stats['chunks']}")
    print(f"  决策总数          : {stats['decisions']}")
    print(f"  中断期间数据块    : {stats['suppress_chunks']}")
    print(f"  中断期间决策      : {stats['suppress_decisions']}")
    print(f"  恢复后决策        : {stats['post_suppress_decisions']}")
    print(f"  N08 自动静止      : {'OK' if stats['auto_stale_ok'] else 'FAIL'}")
    print(f"  停止后反馈关闭    : {'OK' if win.stickman._feedback is False else 'FAIL'}")
    print(f"  状态样本(前8)     : {[str(s) for s in stats['states'][:8]]}")

    # ---- item 2 接入校验（不弹真实对话框，仅验证逻辑）----
    intake_problems = intake_check()
    if intake_problems:
        print("==== 接入校验检查 ====")
        for p in intake_problems:
            print(f"  - {p}")

    if cov_problems:
        problems.extend(cov_problems)
    if intake_problems:
        problems.extend(intake_problems)
    if problems:
        print("\n!! 失败项:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\n✅ 全部通过")
    return 0


class _BrokenChunkSource:
    """测试用：底层返回了非 EEGChunk 的裸数组，模拟"格式错误"路径。"""

    info = SourceInfo(
        source_id="BAD",
        recording_id="X",
        sample_rate_hz=160.0,
        channel_names=tuple(DEFAULT_CHANNELS),
        reference=None,
        subject_id=None,
        run=None,
        n_samples=None,
    )

    def read_chunk(self):
        return np.zeros(8, dtype=np.float64)  # 1D 裸数组，不是 EEGChunk

    def __enter__(self) -> "_BrokenChunkSource":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def close(self) -> None:
        return None


def intake_check() -> list[str]:
    """同步校验 item 2 的接入判定逻辑（不弹真实对话框，只验证 check_source_contract）。"""
    problems: list[str] = []
    good = check_source_contract(MockEegSource())
    if not good.ok:
        problems.append(f"正常源被判为格式错误: {good.message}")
    bad = check_source_contract(_BrokenChunkSource())
    if bad.ok:
        problems.append("item 2 失效：类型/形状错误的源未被判为格式错误（报错框不会弹）")
    return problems
    print("\n✅ 全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
