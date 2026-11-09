# 数据目录

原始 EEG 文件**不入库**：`.gitignore` 忽略 `*.edf` 与 `*.edf.event`。仓库里只保存数据说明、清单、校验值和固定划分。

| 路径 | 内容 |
| --- | --- |
| [`eegmmidb/README.md`](eegmmidb/README.md) | 数据集说明、标签、采集参数、本地缓存范围、下载与校验方法 |
| [`eegmmidb/download_manifest.json`](eegmmidb/download_manifest.json) | 30 个 EDF 的相对路径、大小、SHA-256、官方 URL |
| [`eegmmidb/SHA256SUMS.txt`](eegmmidb/SHA256SUMS.txt) | `sha256sum` 兼容的校验文件 |
| [`../training/splits.json`](../training/splits.json) | S001–S010 的 train / validation / test 映射（run 4 / 8 / 12） |

换机器时：

```powershell
.\.venv\Scripts\python.exe scripts\download_eegmmidb.py --subjects 1-10 --runs 4,8,12
.\.venv\Scripts\python.exe scripts\verify_eegmmidb.py
```

EDF 下载到 `data/eegmmidb/Sxxx/` 后仍被 git 忽略；清单和划分会随仓库走，因此数据和实验划分可以复现。
