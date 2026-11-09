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
7. 存档状态显示（D 的 JsonlResultSink.status 语义）：
   - 状态映射（opening/writing/closing/closed±completed/failed）逐条校验；
   - 保存完成时显示会话目录；
   - 保存未完成（completed=false）显示"保存未完成"；
   - 保存失败时显示原因、动作反馈立即停止、会话停止、并通知 A（signal + SessionControl）。

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

try:
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import QTimer
except ModuleNotFoundError as _exc:  # 最常见的跑错解释器：系统 python 没装 PySide6
    raise SystemExit(
        "⚠ 当前解释器里没有 PySide6：\n"
        f"  {sys.executable}\n\n"
        "本仓库的界面依赖装在 ./../.venv 虚拟环境里，请用它的 python 运行：\n"
        "  cmd      : ..\\.venv\\Scripts\\python.exe examples\\smoke_ui_offscreen.py\n"
        "  git bash : ../.venv/Scripts/python.exe examples/smoke_ui_offscreen.py\n\n"
        f"原始错误：{_exc}"
    )

import numpy as np

from eeg_core.contracts import DecisionState, EEGChunk, SourceInfo
from eeg_app.ui.main_window import MainWindow
from eeg_app.ui.intake import check_source_contract
from eeg_app.ui.save_status import failed_sink, format_sink_status
from mock_source import MockEegSource, DEFAULT_CHANNELS
from mock_decoder import MockDecoder
from mock_sink import MockResultSink


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

    # ---- 存档状态验收（映射 + 生命周期）----
    save_map_problems = save_status_mapping_check()
    save_live_problems, save_info = save_status_live_check(app)

    print("==== 存档状态验收 ====")
    print(f"  closing 显示      : {save_info.get('closing_text')!r}")
    print(f"  完成显示          : {save_info.get('completed_text')!r}")
    print(f"  目录显示          : {save_info.get('dir_label')!r}")
    print(f"  测试模式标注      : {save_info.get('test_mode_label')!r}")
    print(f"  未完成显示        : {save_info.get('incomplete_text')!r}")
    print(f"  失败显示          : {save_info.get('failed_text')!r}")
    print(f"  启动前失败显示    : {save_info.get('early_text')!r}")
    print(f"  失败前动作反馈    : {save_info.get('feedback_before_fail')}")
    print(f"  失败后动作反馈    : {save_info.get('feedback_after_fail')}")
    print(f"  失败后会话停止    : {'OK' if not save_info.get('running_after_fail') else 'FAIL'}")
    print(f"  通知 A 的控制调用 : {save_info.get('control_calls')}")

    if cov_problems:
        problems.extend(cov_problems)
    if intake_problems:
        problems.extend(intake_problems)
    problems.extend(save_map_problems)
    problems.extend(save_live_problems)
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


# ---------------- 存档状态（D 的 JsonlResultSink.status 语义） ----------------
class _RecordingControl:
    """模拟 A 的会话控制接口，记录界面调用过哪些控制动作。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def request_pause(self, reason: str) -> None:
        self.calls.append(("pause", reason))

    def request_stop(self, reason: str) -> None:
        self.calls.append(("stop", reason))


def save_status_mapping_check() -> list[str]:
    """同步校验：status 字典 → 界面文案的映射（纯函数，五条映射全覆盖）。"""
    problems: list[str] = []
    cases = [
        (None, "存档模块未接入", "muted"),
        ({"state": "opening"}, "正在准备保存目录", "info"),
        ({"state": "writing"}, "正在保存", "info"),
        ({"state": "closing"}, "正在写完剩余记录", "info"),
        ({"state": "closed", "completed": True}, "保存完成", "ok"),
        ({"state": "closed", "completed": False}, "保存未完成", "warn"),
        ({"state": "failed", "error": "磁盘已满"}, "保存失败：磁盘已满", "error"),
    ]
    for raw, expect_text, expect_level in cases:
        st = format_sink_status(raw)
        if expect_text not in st.text:
            problems.append(f"状态映射错误：{raw} → {st.text!r}，应包含 {expect_text!r}")
        if st.level != expect_level:
            problems.append(f"状态级别错误：{raw} → {st.level!r}，应为 {expect_level!r}")

    # submit() 只表示入队，绝不能据此显示“保存完成”
    sink = MockResultSink(state="writing")
    sink.submit({"decision": "…"})
    if "保存完成" in format_sink_status(sink.status).text:
        problems.append("submit() 入队被误判为保存完成（应以 status.state 为准）")
    return problems


def save_status_live_check(app: QApplication) -> tuple[list[str], dict]:
    """驱动真实窗口走完存档生命周期（第二个事件循环）。

    覆盖：目录显示、正常保存完成、保存未完成、保存失败 + 失败后动作停止 + 通知 A。
    使用 ``MockResultSink``（模拟存档），**不代表真实链路已接通**。
    """
    problems: list[str] = []
    info: dict[str, object] = {}
    session_dir = os.path.join(_ROOT, "results_ui_test", "session_demo-sink-check")

    def _new_window(sink, control=None) -> MainWindow:
        src = MockEegSource()
        dec = MockDecoder(sample_rate_hz=src.info.sample_rate_hz)
        w = MainWindow(
            source=src, decoder=dec, project_dir=_ROOT,
            result_sink=sink, session_control=control, test_mode=True,
        )
        w.hide()
        w._on_start()
        return w

    sink_a = MockResultSink(state="writing")
    sink_b = MockResultSink(state="writing")
    sink_c = MockResultSink(state="writing")
    control_c = _RecordingControl()
    failed_reasons: list[str] = []

    win_a = _new_window(sink_a)
    win_b = _new_window(sink_b)
    win_c = _new_window(sink_c, control_c)
    win_c.sink_failed.connect(failed_reasons.append)

    # D 的 JsonlResultSink 构造失败时直接抛异常，A 用 failed_sink() 交给界面
    early_control = _RecordingControl()
    early_reasons: list[str] = []
    src_e = MockEegSource()
    dec_e = MockDecoder(sample_rate_hz=src_e.info.sample_rate_hz)
    win_e = MainWindow(
        source=src_e, decoder=dec_e, project_dir=_ROOT,
        result_sink=failed_sink("FileExistsError: 输出目录不可用"),
        session_control=early_control, test_mode=True,
    )
    win_e.hide()
    win_e.sink_failed.connect(early_reasons.append)

    def step_early_failure() -> None:
        info["early_text"] = win_e.save_status.text
        info["early_calls"] = list(early_control.calls)
        if "保存失败" not in info["early_text"] or "输出目录不可用" not in info["early_text"]:
            problems.append(f"启动前存档失败未显示：{info['early_text']!r}")
        if not early_reasons:
            problems.append("启动前存档失败未发出 sink_failed 信号")
        if win_e.stickman._feedback is not False:
            problems.append("启动前存档失败时动作反馈未处于停止状态")

    def step_closing() -> None:
        # writing → closing
        sink_a.set_state("closing", directory=session_dir)
        win_a._poll_sink_status()
        info["closing_text"] = win_a.save_status.text
        if "正在写完剩余记录" not in info["closing_text"]:
            problems.append(f"closing 显示错误：{info['closing_text']!r}")

    def step_closed_completed() -> None:
        # closing → closed(completed=True)：保存完成 + 会话目录
        sink_a.set_state("closed", directory=session_dir, completed=True)
        win_a._poll_sink_status()
        info["completed_text"] = win_a.save_status.text
        dir_label = win_a.save_status_widget._dir_label.text()
        info["dir_label"] = dir_label
        info["test_mode_label"] = win_a.save_status_widget._text_label.text()
        if "保存完成" not in info["completed_text"]:
            problems.append(f"保存完成显示错误：{info['completed_text']!r}")
        if session_dir not in dir_label:
            problems.append(f"保存完成后未显示会话目录：{dir_label!r}")
        if "界面测试" not in str(info["test_mode_label"]):
            problems.append("模拟存档未在界面上标注“界面测试·模拟”，有冒充真实链路的风险")
        win_a._on_stop()

    def step_closed_incomplete() -> None:
        # closed(completed=False)：保存未完成
        sink_b.set_state("closed", directory=session_dir, completed=False)
        win_b._poll_sink_status()
        info["incomplete_text"] = win_b.save_status.text
        if "保存未完成" not in info["incomplete_text"]:
            problems.append(f"保存未完成显示错误：{info['incomplete_text']!r}")
        win_b._on_stop()

    def step_failed() -> None:
        # 失败前：动作反馈应处于激活状态（证明“失败后停止”确实生效）
        info["feedback_before_fail"] = win_c.stickman._feedback
        if not win_c.stickman._feedback:
            problems.append("失败前动作反馈未激活，无法验证失败后是否停止")
        sink_c.set_state("failed", error="磁盘已满", directory=session_dir)
        win_c._poll_sink_status()
        info["failed_text"] = win_c.save_status.text
        info["feedback_after_fail"] = win_c.stickman._feedback
        info["control_calls"] = list(control_c.calls)
        if "保存失败" not in info["failed_text"] or "磁盘已满" not in info["failed_text"]:
            problems.append(f"保存失败提示错误：{info['failed_text']!r}")
        if win_c.stickman._feedback is not False:
            problems.append("存档失败后动作反馈未立即停止")
        if not failed_reasons:
            problems.append("存档失败未发出 sink_failed 信号（A 无法感知）")
        if not any(kind == "stop" for kind, _ in control_c.calls):
            problems.append("存档失败未调用 SessionControl.request_stop")

    def finish() -> None:
        info["running_after_fail"] = win_c._running
        if win_c._running:
            problems.append("存档失败后会话未停止（_running 仍为 True）")
        info["worker_after_fail"] = win_c.worker is not None
        if win_c.worker is not None:
            problems.append("存档失败后回放线程未回收")
        for w in (win_a, win_b, win_c, win_e):
            w.close()
        app.quit()

    QTimer.singleShot(500, step_closing)
    QTimer.singleShot(650, step_closed_completed)
    QTimer.singleShot(800, step_closed_incomplete)
    QTimer.singleShot(1100, step_failed)
    QTimer.singleShot(1200, step_early_failure)
    QTimer.singleShot(1400, finish)
    app.exec()
    return problems, info


if __name__ == "__main__":
    raise SystemExit(main())
