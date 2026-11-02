# EEGMMIDB 数据说明（PhysioNet EEG Motor Movement/Imagery Dataset v1.0.0）

本目录缓存项目使用的原始 EDF 文件。**EDF 不入库**（`.gitignore` 忽略 `*.edf` 和 `*.edf.event`），仓库只保留本说明、`download_manifest.json`、`SHA256SUMS.txt` 和 `training/splits.json`。

## 1. 数据集身份

| 项目 | 值 |
| --- | --- |
| 名称 | EEG Motor Movement/Imagery Dataset |
| 版本 | 1.0.0（2009-09-09） |
| 作者 | Gerwin Schalk |
| DOI | [10.13026/C28G6P](https://doi.org/10.13026/C28G6P) |
| 数据页 | <https://physionet.org/content/eegmmidb/1.0.0/> |
| 许可证 | [Open Data Commons Attribution License v1.0](https://physionet.org/content/eegmmidb/view-license/1.0.0/)（ODC-By 1.0） |

使用数据时请引用：

> Schalk, G. (2009). EEG Motor Movement/Imagery Dataset (version 1.0.0). PhysioNet. RRID:SCR_007345. https://doi.org/10.13026/C28G6P

以及原始 BCI2000 论文：Schalk et al., *IEEE Transactions on Biomedical Engineering* 51(6):1034–1043, 2004.

## 2. 本地缓存范围

| 项目 | 值 |
| --- | --- |
| 受试者 | S001–S010（10 人；完整数据集为 S001–S109） |
| Run | 4、8、12（单侧拳头运动想象：左手 / 右手） |
| EDF 文件 | 30 个（每人 3 个 run） |
| 附属文件 | 30 个 `.edf.event`（PhysioNet 生成的二进制注释；本项目不使用） |
| 体积 | 30 个 EDF 共 77,040,000 字节（约 73.5 MiB）；含 `.event` 约 77.1 MB |
| 采集参数 | 64 通道、160 Hz、每个 run 123 s 或 125 s |
| 标注来源 | EDF+ 内嵌 annotation channel（T0/T1/T2），无需解析 `.edf.event` |

## 3. 目录结构

```text
data/eegmmidb/
├── README.md                  # 本文件（入库）
├── download_manifest.json     # 30 个 EDF 的相对路径 / size / sha256 / url（入库）
├── SHA256SUMS.txt             # 30 行 sha256sum 格式（入库）
├── S001/
│   ├── S001R04.edf            # run 4，训练
│   ├── S001R08.edf            # run 8，验证
│   ├── S001R12.edf            # run 12，封存测试
│   └── S001R04.edf.event …    # 可选，本地缓存，不入库
├── S002/ … S010/
```

## 4. 标签与时序

Run 4 / 8 / 12 是**单侧拳头运动想象**。每个 run 有 15 个 trial，标注为：

| 标签 | 含义（仅 run 4/8/12） | 时长 |
| --- | --- | --- |
| `T0` | 静息（rest） | 4.1 s 或 4.2 s |
| `T1` | 左手拳头运动想象 | 4.1 s |
| `T2` | 右手拳头运动想象 | 4.1 s |

每个 run 约 15 个 `T0`、7–8 个 `T1`、7–8 个 `T2`；3 个 run 合计约 45 个 trial。注意：`T1` / `T2` 的含义随 run 类型变化——在 run 5/6/9/10/13/14 的双侧任务中，`T1` 是双手、`T2` 是双脚；本项目只启用 4/8/12，不能把其他 run 的同名标签直接沿用。

时序细节（2026-11-02 在本机 30 个文件上核对）：

- 123 s 文件：`T0` 为 4.1 s，最后一个 trial 在 123.0 s 结束；
- 125 s 文件：`T0` 为 4.2 s，最后一个 trial 在 124.5 s 结束，尾部 0.5 s **没有标注**，不能默认当作 idle；
- `T0/T1/T2` 的起点不是 40 样本（250 ms）网格的整数倍：onset 对 40 取余会循环出现 `0, 8, 16, 24, 32`。因此窗口起点、0.5 s 保护区、1 s 评估排除区都必须按**样本索引**计算，不能按秒四舍五入。

## 5. 通道名说明

EDF 头里的通道名是 BCI2000 风格，大小写混合且带尾点，例如：

```text
Fc5.  Fc3.  Fc1.  Fcz.  Fc2.  Fc4.  Fc6.  C5..  C3..  C1..  Cz..  C2..  C4..  C6..
```

模型通道在契约里写作 `FC3`、`FCz`、`FC4`、`C3`、`Cz`、`C4`、`CP3`、`CPz`、`CP4`。读取层不能用 `raw.ch_names` 直接和 `MODEL_CHANNELS` 做等值比较，需要一张显式别名表（例如 `Fc3.` → `FC3`、`C3..` → `C3`、`Cpz.` → `CPz`），并保持行顺序。该映射目前仍是待实现项，不能假设 `strip('.').upper()` 就够（`FCz`、`Cz`、`CPz` 会失败）。

## 6. 完整性与校验

`download_manifest.json` 记录每个 EDF 的相对路径、字节数、SHA-256 和官方 URL；`SHA256SUMS.txt` 是同一批哈希的 `sha256sum` 格式副本。当前本地 30/30 个 EDF 的大小与 SHA-256 均与清单一致。

```powershell
# 校验（默认检查大小 + SHA-256；--mne 额外检查 EDF 头与标注）
.\.venv\Scripts\python.exe scripts\verify_eegmmidb.py
.\.venv\Scripts\python.exe scripts\verify_eegmmidb.py --mne
```

## 7. 下载 / 扩充

```powershell
# 30 个 EDF（默认 S001–S010、run 4/8/12）
.\.venv\Scripts\python.exe scripts\download_eegmmidb.py --subjects 1-10 --runs 4,8,12

# 全数据集（109 人；约 3.4 GB，按需使用）
.\.venv\Scripts\python.exe scripts\download_eegmmidb.py --subjects 1-109 --runs 4,8,12
```

脚本优先走 AWS 公开镜像 `physionet-open.s3.amazonaws.com`，失败后回落官网 `physionet.org/files/eegmmidb/1.0.0/`；已存在且 SHA-256 匹配的文件会跳过，新增文件会写回清单。网络慢时可先设置 `HTTPS_PROXY`（脚本使用 Python 标准库，会自动读取代理环境变量）。

## 8. 固定划分

`training/splits.json` 固定每人：

- run 4 → train（拟合模型与所有需要学习的变换）
- run 8 → validation（选通道、门限、时间规则）
- run 12 → sealed test（冻结后一次性评估）

开发受试者为 S001–S005；正式集合为 S001–S010。所有受试者使用同一选择规则；重叠窗口不能随机划分到训练和测试两边。
