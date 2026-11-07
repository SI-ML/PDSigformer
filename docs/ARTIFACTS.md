# 数据与产物

```text
artifacts/
  objects/sha256/<前两位>/<SHA>.pt 或 .pth
  manifests/baseline78_unified_v4.json
  manifests/pds24.json
  data_snapshots/pds_shimmer/
  data_snapshots/pds_pads/
  data_snapshots/baseline_common/prepared/<task>/
  original_evidence/...
  commands/                 历史数值兼容器
  native_compat/ 与清单列出的运行库目录
  IMPORT_RECEIPT.json
```

导入脚本只复制必要文件并逐一验证 SHA，不修改来源。PDS 归档缺失的 Train/Valid 数组，仅在基线归档文件 SHA、样本身份、顺序全部一致时恢复，来源记录在 IMPORT_RECEIPT。

PDS 缓存包含 manifest.json、train_X.npy / train_y.npy / train_windows.csv，以及 valid/test 的同类文件。X 为 float32 `[N,T,C]`，y 为 int64 `[N]`。CSV 含 loader_index、label、subject_id、window_id 等冻结身份字段。基线缓存存为 `<split>/X.npy`、y.npy、identities.csv，任务说明为 task.json。

源码不含数据。原始数据的分窗、类别筛选和受试者划分须使用已归档预处理材料；相同数组形状不代表相同数据。必须提供文件摘要和身份对应。训练只需要 Train/Valid 缓存，不需要旧权重。

## 历史推理运行库

TimesNet/Autoformer 历史 Test 使用 cuFFT 10.9 数值路径，Reformer 使用冻结 CUDA 11.8 GELU PTX，统一 Torch 2.8.0+cu128。文件列于 baseline manifest 的 runtime_files 中，验证 SHA 后加载。运行库从实验产物包导入，不放入 Git 源码；Reformer 当前验收 GPU 为 SM89。

## 记录范围

公开源码保留配置、SHA、表格映射、聚合指标与测试状态。权重、数据、逐样本身份和预测、服务器路径及完整原始凭证留在产物目录。checkpoints/ 提供 108 个可读引用，对应 102 份唯一对象。outputs/ 每次执行新建目录，失败尝试不被覆盖。
