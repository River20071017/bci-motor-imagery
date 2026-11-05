# BCI Competition IV 2a（BNCI 2014-001）数据说明

## 1. 数据集身份

| 项目 | 值 |
| --- | --- |
| 名称 | BCI Competition IV Dataset 2a（BNCI Horizon 2014-001） |
| 内容 | 9 名受试者 × 2 个 session（T/E）× 6 run × 48 trial；4 类运动想象（左/右/双脚/舌） |
| 采集 | 22 导 EEG + 3 导 EOG，250 Hz，0.5–100 Hz 带通 + 50 Hz 陷波，左乳突参考 |
| 论文 | Tangermann et al., *Frontiers in Neuroscience* 6:55, 2012，[DOI](https://doi.org/10.3389/fnins.2012.00055) |
| 许可证 | CC BY-ND 4.0 |
| 文件 | 每人 2 个 `.mat`（`A01T.mat` / `A01E.mat`），本目录缓存，**不入库** |

## 2. 分类与窗（本项目的映射）

| 本项目的类 | 2a 来源 | 说明 |
| --- | --- | --- |
| `idle`（`T0`） | 上一段想象结束 → 本次提示出现 | 含 run 间休息与 2 s 注视期；连续区间足够放 2 s 窗 |
| `left_hand`（`T1`） | class 1 的想象段，`[提示 + 1 s, 提示 + 4 s)` | 跳过提示呈现/视觉加工 |
| `right_hand`（`T2`） | class 2 的想象段，同区间 | — |
| `T3` / `T4` | class 3/4（双脚/舌）的想象段 | **保留 marker 但不映射**，`label_of_window` 返回 `None`，窗自动排除 |

- 窗长 2 s、步长 250 ms（250 Hz → 500 / 62 样本）；
- 每个 session 约 **1000 个可用窗**（左右各约 360、idle 约 288），约为 EEGMMIDB 的 4 倍；
- 算法配置用 `AppConfig(source=SourceConfig(sample_rate_hz=250.0))`，**不重采样**。

## 3. 训练 / 验证 / 测试

| 部分 | 数据 | 用法 |
| --- | --- | --- |
| 训练 + 验证 | session **T**（6 run） | **按 run 分组的 6 折交叉验证**：每折留 1 个 run 验证、5 个 run 训练；超参/特征定完后用全部 6 个 run 重训最终模型 |
| 测试 | session **E**（6 run） | **封存**：加载层默认拒绝，只有 `allow_sealed=True` 才可读；只在冻结后跑一次 |

- 划分文件：[`training/splits_bnci2014.json`](../../training/splits_bnci2014.json)；代码：[`training/cv.py`](../../training/cv.py)；
- 不按 trial 或窗随机划分——重叠窗会泄漏；run 是采集上独立的块；
- 每人一个模型，与 EEGMMIDB 的 per-subject 方案一致；模型放在 `training/models/bnci2014/`（不入库）。

## 4. 下载

```powershell
# 9 人 × T/E（约 500 MB）；镜像不可达时加 --insecure 或手动下载
.\.venv\Scripts\python.exe scripts\download_bnci2014.py --subjects 1-9 --sessions T,E
```

脚本按顺序尝试 TU Graz 镜像与 BNCI Horizon 原站，校验文件大小与 MAT 结构，并写
`download_manifest.json` / `SHA256SUMS.txt`。如果镜像不可达，也可以用 MOABB 下载后把文件
放进本目录，再重跑脚本做校验。

## 5. 训练与评估

```powershell
# 只做 CV + 全 T 重训（不碰 E）
.\.venv\Scripts\python.exe scripts\train_bnci2014.py --subjects 1-9

# 冻结后，一次性评估 E（跑完不要再改任何东西）
.\.venv\Scripts\python.exe scripts\train_bnci2014.py --subjects 1-9 --allow-sealed
```

## 6. 与 EEGMMIDB 的差异

| 项 | EEGMMIDB | 2a |
| --- | --- | --- |
| 采样率 | 160 Hz | **250 Hz**（独立配置与模型） |
| 通道数 | 64 | 22 EEG + 3 EOG |
| 类别 | 3 类（T0/T1/T2） | 4 类，只用左/右 + 注视期 idle |
| 划分 | run4/run8/run12 | T 内 run CV + T 全量重训 + E 封存 |
| 质量阈值 | 500 µV / 1 s / 1 µV（160 Hz 标定） | 先沿用，接入真实数据后必须用 `scripts/calibrate_quality.py` 重标 |
