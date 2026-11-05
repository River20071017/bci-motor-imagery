# 训练与评估（B 侧）

本目录实现“A 的预处理 → B 的分窗/特征 → 三状态 LDA → 决策”的离线训练链路。
模型文件不入库（`.gitignore` 已忽略 `training/models/*.npz` 与 `*.model.json`）。

分层：`training/pipeline.py` 是纯管线，只吃 `RecordingLike`，不 import `eeg_app`；
`train.py` / `evaluate.py` 是入口，只有 `main()` 里才 `from eeg_app.sources.input import load_edf`。
依赖方向与边界见 [docs/00_架构总览.md](../docs/00_架构总览.md)。

## 1. 前置

```powershell
# 仓库根目录
.\.venv\Scripts\python.exe -m pytest -q          # 当前 81 条
.\.venv\Scripts\python.exe scripts\verify_eegmmidb.py
```

训练/评估读 EDF，需要 `mne`；模型拟合需要 `scikit-learn`（均已在 `requirements.txt`）。

## 2. 训练

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
.\.venv\Scripts\python.exe -m training.train --subjects S001,S002,S003,S004,S005
```

输出到 `training/models/`：

| 文件 | 内容 |
| --- | --- |
| `<model_id>.weights.npz` | `lda_npz_v1`：`coef` / `intercept` / `priors` / scaler 均值与尺度 |
| `<model_id>.model.json` | `ModelInputSpec` + `DecisionConfig` + 训练数据统计 + 权重 SHA-256 |

`model_id` 形如 `mi-S001-lda-20261103-a844d28`；`pipeline_version` 在
`src/eeg_core/version.py`，改滤波/特征行为时必须同步改。

每人的训练只读 `splits.json` 的 `train`（run 4）；run 8 只用于选门限，run 12 封存。

### 2.1 Mock EEGChunk（不读 EDF）

```powershell
.\.venv\Scripts\python.exe scripts\make_mock_eeg.py `
  --out data\mock\S999R04.mock.npz --duration-sec 24
```

- 64 通道、160 Hz、50 ms 块；T0/T1/T2 标记；T1/T2 在 8–13 / 13–30 Hz 上做左右不对称。
- `--artifact spike|flatline|nan` 注入伪迹，用来测质量门。
- 输出是 chunk 流 `.npz`，不是 EDF；读取：

```python
from training.mock import load_mock_stream

stream = load_mock_stream("data/mock/S999R04.mock.npz")
chunks = list(stream)             # EEGChunk 流
recording = stream.as_recording()  # RecordingLike，可喂 build_feature_dataset
```

## 3. 评估

```powershell
.\.venv\Scripts\python.exe -m training.evaluate `
  --model training\models\mi-S001-lda-20261103-a844d28.model.json `
  --subject S001 --part validation
```

当前是**窗口级**指标：每个 2 s 窗一个预测，不按 250 ms 决策点计分，也不排除标签
转换区。它是开发期基线，不能直接当 PRD §10 的封存测试报告。

## 4. 当前基线（2026-11-03，默认契约配置）

| 受试者 | 可用窗（run8） | 宏 F1 | 左右手平衡召回 | 备注 |
| --- | ---: | ---: | ---: | --- |
| S001 | 206 | 0.448 | 0.551 | |
| S002 | 249 | 0.529 | 0.505 | |
| S003 | **0** | — | — | run8 有 145 个 invalid 块，标准窗 0 |
| S004 | 249 | 0.572 | 0.478 | |
| S005 | 249 | 0.426 | 0.260 | |
| 四人均值 | — | **0.494** | **0.448** | 不含 S003 |

训练集（run4）窗口数：S001 208、S002 249、S003 45（idle 34 / 左 8 / 右 3）、
S004 249、S005 249。S003 的 45 窗是过拟合样本，不要用于正式结论。

**结论**：链路已跑通，但四人均值宏 F1 0.494、左右手平衡召回 0.448，低于 PRD 建议线
（0.50 / 0.60）。下一步先修质量门（S003/S009），再谈特征/模型改进；不要在 S003 的
45 窗上宣布达标。

## 5. 已知卡点

1. **质量门范围**：`QualityChecker` 当前检查全部 64 通道，非模型通道的额区伪迹会把
   整块判 `invalid`；S003/S009 因此几乎/完全没有可用窗。需要 A 决定收到 9 个模型通道
   还是抬阈值；改完必须重训（`preprocessing_hash` 会变，或数据分布会变）。
2. **评估口径**：PRD §10 的决策点评估（排除转换前后 1 s、起始 3 s，uncertain/invalid
   计入遗漏）尚未接到 `evaluation/`；当前数字只用于开发比较。
3. **`Decoder`/`ResultSink`**：未定义；离线训练不受影响，接 UI/存档前要定。

## 6. 代码地图

| 文件 | 职责 |
| --- | --- |
| `src/eeg_core/processing/windows.py` | 2 s / 250 ms 网格、整窗可用判断 |
| `src/eeg_core/processing/features.py` | 18 维对数方差 |
| `training/pipeline.py` | 流式 `FilterBank`、有效段、标签、数据产出统计 |
| `training/mock.py` | mock `EEGChunk` 流、`.npz` 读写、`as_recording()` |
| `scripts/make_mock_eeg.py` | 生成 mock chunk 流的 CLI |
| `src/eeg_core/decoding/models.py` | LDA 训练/打分/权重读写 |
| `src/eeg_core/decoding/inference.py` | 模型包加载与四道校验 |
| `src/eeg_core/decoding/decision.py` | 质量/预热/门限 → `DecisionFrame` |
| `training/train.py` | 按 `splits.json` 训练并导出模型包 |
| `training/evaluate.py` | 窗口级评估 |
