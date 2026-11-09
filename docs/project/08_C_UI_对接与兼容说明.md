# C 的界面改动：对接、兼容与迁移说明

> 适用分支：`feat/C-ui`（4 个提交）。本文件说明：前面已完成的界面改动**对外部代码意味着什么**、
> 外部模块**如何调整才能保持兼容**、以及**把改动扩展/同步到外部代码**时必须遵守的约束与验收标准。

## 0. 一句话结论

- C 的 UI 在**当前 `main` 契约**与 **B 的冻结契约**（`origin/feature/protocol`）下**都能正常导入与运行**（已实测，见 §5）。
- 仓库里除 `src/eeg_core/contracts/`（A 的契约）和 C 的 `src/eeg_app/ui/`、`examples/` 外，**其余模块目前都是 0 字节占位**。
  因此"现在就要改的外部代码"其实没有；真正要做的是两件事：
  1. **契约冻结落地时**，按 §3 的迁移表改 import（只影响"引用了被移动/改名符号"的代码）；
  2. **A/B/D 实现各自模块时**，按 §4 的对接约定来写，保证能插进 C 的 UI。
- C **不修改** `eeg_core/contracts/`（硬边界①：契约只由 A 合并）。

## 1. C 已完成的改动清单

| 提交 | 内容 | 新增/修改文件 | 对外影响 |
|---|---|---|---|
| `8a9613b` | 界面测试版 UI 骨架（波形/时间轴/火柴人/状态/主窗/回放控制器 + mock） | `src/eeg_app/ui/*.py`、`examples/*.py` | 提供可注入 `source`/`decoder` 的 `MainWindow` |
| `e0f998e` | 留后手：UI 不再直接依赖 `contracts.DataSource` 符号 | `playback.py`、`main_window.py`、`mock_source.py` | 新增本地 `SourceProtocol`（结构约定） |
| `afc05fc` | item2：应用层数据格式校验 + 报错/确认弹窗 | `intake.py`（新）、`main_window.py`、`smoke_ui_offscreen.py` | 新增 `check_source_contract()` 纯函数 |
| `8a81162` | 运行过程切屏（思考管线 + 处理日志 + 波形绘制过程） | `process.py`（新）、`main_window.py` | `MainWindow` 改为 `QTabWidget` 双屏 |

**C 的 UI 与外部唯一的"接缝"**：
- 输入：`source`（满足 `SourceProtocol`：`.info` / `.read_chunk()` / 上下文管理器），产出合法的 `EEGChunk`；
- 输出：`decoder`（满足 `DecoderProtocol`：`set_session` / `reset(reason)` / `decode(chunk) -> DecisionFrame | None`）；
- 界面只消费 `EEGChunk`（画波形）与 `DecisionFrame`（火柴人/时间轴/状态/过程屏）。

## 2. 契约变动对照（`main` → 冻结 v1）

| 符号 | 当前 `main` | 冻结 v1（`origin/feature/protocol`） | 外部代码怎么处理 |
|---|---|---|---|
| `DataSource` | `contracts/protocols.py`，且 `contracts.__init__` 再导出 | **已从 contracts 移除** | 不要从 `eeg_core.contracts` 导入；改为"结构约定"（C 已提供 `eeg_app.ui.playback.SourceProtocol`），或由 A 在 `eeg_app/sources/` 内定义 |
| `StreamConsumer` | `protocols.py` | **已移除** | 解码器/滤波器不再继承它；直接用 `reset(reason)` 约定 |
| `MarkerSource` | `protocols.py` | **已移除** | 参考标注只走事件对象 `ReferenceMarker`，不进解码路径 |
| `SourceError` / `SourceClosedError` | `protocols.py` | **已移除**（仅保留 `IncompatibleSourceError`） | 改为普通 `Exception`/自定义异常，或由 A 的 sources 定义 |
| `EEGChunkError` | `frames.py`（异常名） | **改名 `ContractError`** | `except EEGChunkError` → `except ContractError`；兼容写法见 §3.1 |
| `DecisionFrame` / `Scores` / `QualityReport` / `DecisionState` | `frames.py` | 移到 **`outputs.py`**（仍经 `contracts.__init__` 再导出） | `from eeg_core.contracts import DecisionFrame` **仍可用**；只有直接引用子模块时才改路径 |
| `ResetReason` | `protocols.py` | 移到 **`events.py`**（仍经 `contracts.__init__` 再导出） | `from eeg_core.contracts import ResetReason` **仍可用** |
| `ChunkStreamGuard` 方法 | `accept(chunk)` | **`validate(chunk)`** | 逐块校验时用 `getattr(guard, "validate", None) or getattr(guard, "accept", None)`，兼容两版 |
| `ChunkSequenceError` 基类 | `SourceError` | `ContractError` | 只 `except Exception` 或同时兜住两版基类即可 |

**关键点**：冻结契约的 `contracts.__init__` **仍然再导出** `DecisionFrame/DecisionState/Scores/QualityReport/ResetReason/EEGChunk/SourceInfo/...`。
所以外部代码里绝大多数 `from eeg_core.contracts import X` **不用改**；真正会断的只有上表"已移除/改名"的几项（尤其 `DataSource`、`EEGChunkError`）。

## 3. 外部代码按模块如何调整

### 3.1 通用：异常与守卫的兼容写法（建议直接抄）

```python
# 兼容 main(EEGChunkError) 与冻结(ContractError)
try:
    from eeg_core.contracts import ContractError
except ImportError:
    from eeg_core.contracts import EEGChunkError as ContractError

# 兼容 ChunkStreamGuard.accept / validate
guard = ChunkStreamGuard(info)
checker = getattr(guard, "validate", None) or getattr(guard, "accept", None)
if checker is not None:
    checker(chunk)
```

### 3.2 A：数据源与运行时（`eeg_app/sources/*`、`runtime/*`、`app.py`、`__main__.py`）

- **`sources/input.py` / `replay.py`**：产出的对象必须满足 C 的 `SourceProtocol` 结构：
  `info`（`SourceInfo`）、`read_chunk() -> EEGChunk | None`、`__enter__/__exit__`、`close()`。
  首块 `seq=0` 且 `discontinuity=True`；块按 `seq` 递增、`start_sample` 首尾相接；结束返回 `None` 并保持。
- **`runtime/session.py`**：正式集成时**取代** C 的 `examples`/`playback.py`（后者只作测试夹具）。它应：
  循环 `source.read_chunk()` → 把 `EEGChunk` 交给 UI（画波形）→ 调 `decoder.decode(chunk)` → 把 `DecisionFrame` 交给 UI；
  并负责暂停/继续/停止时的 `ResetReason`（`SESSION_START` / `PAUSE_RESUME` / …）。
- **`app.py` / `__main__.py`**：唯一的"组装点"，示意：
  ```python
  from eeg_app.ui.main_window import MainWindow
  source = open_source(path)          # A 提供，满足 SourceProtocol
  decoder = build_decoder(model_dir) # B 提供，满足 DecoderProtocol
  win = MainWindow(source=source, decoder=decoder)
  win.show()
  ```
  （函数名为示意，以 A/B 实际命名为准。）**C 的 UI 不需要改**，只要注入这两个对象。

### 3.3 B：解码与处理（`eeg_core/decoding/*`、`processing/*`）

- 解码器实现 `DecoderProtocol`：`set_session(session_id)`、`reset(reason)`、`decode(chunk)`。
  `decode` 返回 `DecisionFrame | None`；发布节奏由解码器决定（PRD 6.1，约 250 ms 一次）。
- 产出必须能通过 `DecisionFrame` 构造校验：`warming_up`/`invalid` **不带 scores**；`left/right` 必须 `intent==state`；
  `invalid` 必须 `quality.valid=False` 且给原因；`valid_for_ms` 默认 750。
- 处理器（滤波/加窗/特征/质量）实现 `reset(reason)` 语义即可；不再需要继承 `StreamConsumer`（冻结后已移除）。

### 3.4 D：存档与输出（`eeg_app/outputs/*`）

- 只**消费** `DecisionFrame`（必要时 `EEGChunk`）写盘/统计；**不得修改** frozen 数据盒（构造后不可变）。
- **不得**把 `ReferenceMarker`/参考标签喂回解码路径（标签隔离）。

### 3.5 测试与评估（`tests/*`、`evaluation/*`、`training/*`）

- 凡 `from eeg_core.contracts import DataSource / StreamConsumer / MarkerSource / SourceError / SourceClosedError / EEGChunkError`
  的地方，按 §2 迁移表逐条改；其余 import 不动。
- 涉及 `ChunkStreamGuard` 的测试，按 §3.1 的兼容写法。

## 4. 必须遵守的约束（硬边界）

1. **契约只由 A 合并**：任何人（含 C）不直接改 `eeg_core/contracts/`。
2. **分工边界**：数据源=A，算法=B，界面=C，存档=D；跨层只通过契约数据盒通信。
3. **界面只渲染 `DecisionFrame`**：UI 不改写结果；**动画不得由参考标签触发**（`timeline.py` 两轨独立）。
4. **在线模式参考轨必须为空**：`TimelineWidget(reference_mode=False)`；参考标注在解码路径**不可访问**（PRD 5.3 / T03）。
5. **节奏约定**：数据块约 50 ms；决策约 250 ms；`valid_for_ms=750`（超时动作反馈自动静止，N08；暂停/停止 <250ms 停止反馈，N07）。

## 5. 预期输出 / 验收方式

1. **离屏冒烟测试**（无需显示器）：
   ```bash
   python examples/smoke_ui_offscreen.py     # 期望末尾 “✅ 全部通过”，EXIT=0
   ```
   覆盖：6 种决策状态的契约校验、N07、N08、item2 接入校验（正常源通过 / 坏源判错）、
   存档状态五条映射 + 目录显示 + 失败后动作停止 + 通知 A（见 §8）。
2. **冻结契约导入校验**（已验证通过）：
   把 `origin/feature/protocol` 的 `src/eeg_core/contracts` 抽到临时目录并**置于 `sys.path` 最前**，
   再导入 C 的全部 UI 模块并跑一轮解码。期望：`FROZEN_IMPORT_OK`，且能产出合法 `DecisionFrame`。
   （本次实测：`intake ok: True`，冻结契约下产出 80 条决策，全部通过校验。）
3. **集成后**：把真实 `source`/`decoder` 注入 `MainWindow`，波形滚动、决策出现、过程屏有日志与绘制动画；
   在线模式参考轨为空。**C 的 UI 代码不需要改**。

## 6. 把改动扩展/同步到外部代码的步骤

1. 等 A 的 `sources`/`runtime`、B 的 `decoding`、D 的 `outputs` 实现落地（当前均为占位）。
2. 按 §3.2 的 `app.py` 组装：注入真实 `source`/`decoder`，**不碰 UI**。
3. 按 §2 迁移表统一 import（尤其 `DataSource`、`EEGChunkError`、`ChunkStreamGuard.*`）。
4. 跑 §5 的两项验收；任何一项失败，优先检查"符号是否还在 `contracts.__init__` 再导出"。
5. 若契约再次变动：**先**在 `docs/01_接口约定_1.0.md` 与 `contracts` 同步版本号，**再**改本文件 §2/§3。

## 7. 分支与上传

- 存档状态接入分支：`feat/C-ui-save-status`（基于 `origin/main`）。
- 早期交接分支：`feat/C-ui-integration`（= 代码 + 本说明）。
  ```bash
  git push -u origin feat/C-ui-save-status
  ```
- 提醒：提交作者用本人身份（仓库内已设 `Wsz917 <Sizheng0917@163.com>`）；若沿用默认身份，上传前可改：
  ```bash
  git commit --amend --author="王斯正 <你的邮箱>"
  ```

## 8. 存档状态接入（D 的 `JsonlResultSink`）与 A 的交接

D 的存档模块（PR #11 / 分支 `feat/D-session-archive`，尚未合并 main）提供
`JsonlResultSink.status`，字段为 `state / directory / error / queued / completed`。
C 的界面**只读**这个字典并显示，不构造 sink、不 `submit()`、不 `close()`、不写文件。

### 8.1 状态 → 界面文案（`src/eeg_app/ui/save_status.py`）

| `status.state` | 条件 | 界面显示 | 颜色 |
| --- | --- | --- | --- |
| `opening` | — | 正在准备保存目录… | 蓝 |
| `writing` | — | 正在保存… | 蓝 |
| `closing` | — | 正在写完剩余记录… | 蓝 |
| `closed` | `completed=true` | 保存完成 + 会话目录 | 绿 |
| `closed` | `completed=false` | 保存未完成 | 橙 |
| `failed` | — | 保存失败：`<error>` | 红 |
| —（无 sink） | — | 存档模块未接入 | 灰 |

- 轮询周期 400 ms（`SINK_POLL_MS`），只做一次字典拷贝；**`submit()` 成功 ≠ 保存完成**。
- 后台写入可能稍后失败，所以界面持续轮询，而不是在 `submit()` 返回时宣布成功。
- 存档失败时界面立即：停止动作反馈（`mark_stale`）→ 停止本会话回放 → 发 `sink_failed` 信号
  → 调 `SessionControl.request_stop(...)` → 弹**非模态**提示框。界面不自行补写文件。
- 界面测试模式下状态条前缀为「存档（界面测试·模拟）」，避免把模拟存档当真。

### 8.2 A 必须提供的接口（C 侧已完成，等 A 接上）

1. **注入 sink**：会话开始时在工作线程构造 `JsonlResultSink(root, session_id, metadata)`
   （构造会等目录初始化，**不能放 GUI 主线程**），构造成功后回到主线程调用
   ```python
   window.attach_result_sink(sink)     # 主线程；跨线程请用 Qt 队列信号投递
   ```
   或直接在组装时作为构造参数传入：`MainWindow(..., result_sink=sink, test_mode=False)`。
2. **喂数据**：每个 `DecisionFrame` 由 A（或 D 的适配层）调用 `sink.submit(frame)`；
   C 的界面不参与，也不知道队列长度。
3. **收尾**：会话结束/停止时由 A 在工作线程调用 `sink.close(completed=..., reason=...)`，
   不要由界面调用；`close()` 返回后 `status` 变 `closed`，界面自动显示「保存完成 / 保存未完成」。
4. **控制接口**（存档失败时界面用）：实现其一即可
   - `SessionControl`：`request_pause(reason)` / `request_stop(reason)`，通过
     `MainWindow(..., session_control=ctrl)` 注入；
   - 或连接信号 `MainWindow.sink_failed(str)`，在 A 的运行时里做收尾。
   两者可同时使用（信号用于解耦观察，控制接口用于直接叫停），界面不会重复触发。
5. **不要**让 sink 在 `writing` 之前就被 attach，也不要在 `close()` 之后再 `submit()`
   （D 的实现会让 `submit` 抛错并使 sink 永久 `failed`）。

### 8.3 时序（A 的组装入口）

```
工作线程: new JsonlResultSink(...)      → status: opening → writing
        → 回主线程 attach_result_sink(sink) → 界面显示「正在保存…」
主循环  : 每帧 sink.submit(df)           → 界面轮询 queued（仅展示，不判定完成）
收尾    : 工作线程 sink.close(completed=True, reason="finished")
        → status: closing → closed      → 界面显示「正在写完剩余记录…」→「保存完成」+ 目录
异常    : status: failed(error=...)      → 界面显示「保存失败：…」+ 停动作 + 通知 A
```

### 8.4 待联调项（C 无法独立完成）

- [ ] A 在 `eeg_app/app.py` 组装入口提供 sink 的构造、attach、close 三处调用。
- [ ] A 提供 `SessionControl` 实现（或确认改用 `sink_failed` 信号）。
- [ ] 真实 `JsonlResultSink` 的 `status` 字段名/取值与本文档一致（当前按 D 的 PR #11 实现编写）。
- [ ] D 的 PR #11 合并到 main 后，把 `examples/mock_sink.py` 换成真实 sink 再跑一遍验收。
- [ ] 真实会话目录权限/磁盘满场景验证（当前用模拟 `failed` 覆盖）。

