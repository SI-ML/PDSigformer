# 固定论文协议

本次整理对应 13 baseline + PDSigformer 主实验和 R1–R4 消融，两个数据集各三个模型随机种子。受试者分区固定为 split42。

| 角色 | TimesNet 与残差 | 多粒度 | 粒度间交互 |
|---|---|---|---|
| R1 | 无 | 无 | 无 |
| R2 | 有 | 无 | 无 |
| R3 | 有 | 有 | 无 |
| R4 / PDSigformer | 有 | 有 | 有 |

各行均保留粒度内注意力。R1/R2 单粒度为 Shimmer `2`、PADS `32`。R3/R4 的多粒度配置见 [configs/paper.json](../configs/paper.json)。R1 保持 R2 构造后提取 backbone 的初始化；R3 保持 R4 构造后移除 inter_attention 的初始化。

PDS 后端为 main `891f65b...`；历史 Medformer 及原生基线为论文版 `446275f...`。旧版仅模型与必要层单独命名，不覆盖 main。FEDformer 非张量随机频率索引用 `construction_metadata` 保存和恢复。权重严格匹配全部键；PDS 历史推理另外核验初始化 SHA，包括原见证的固定位置编码 buffer。

## 数据与指标

- HC=0，PD=1；输入 `[B,T,6]`。
- 按每窗口每通道沿时间维 StandardScaler，已有补零包括在标准化中。
- 缓存不再次标准化、裁剪、分窗或补零；保持原评估 batch、顺序与全 1 mask。
- Accuracy 使用 argmax；Precision、Recall、F1 为 macro；AUROC、AUPRC 为两类 one-vs-rest macro；AUPRC 使用 `average_precision_score`。
- 三个 seed 的表格为均值与**样本标准差 ddof=1**；重复推理不重复计入均值。

## 训练

PDS 使用 Adam，TimesNet 基线使用 RAdam，其他基线使用 Adam；weight_decay=0、恒定学习率、无权重 CE、梯度裁剪 4。Medformer 基线每轮更新 SWA 并用其 Valid 分数选模，PDS 不用 SWA。

选择 delta=1e-5、预算 100、patience=100；仍保存 Valid-best，并非最后一轮。基线评估随机种子由 seed/model/task/split 派生，评估后恢复训练 RNG。PDS 历史 Test 保持“设 seed → 构造 → 载入权重 → Test”的顺序。

训练只加载 Train/Valid，Test 通过独立 eval 命令执行。新入口复用官方分类基类、模型接口和损失，独立实现历史训练、SWA、选模及日志流程。

## 验收边界

推理复现要求权重、结构、数据、顺序、batch 与运行库对应。FFT/GELU 数值兼容只用于历史 Test，训练使用可微的标准 PyTorch 算子。改变底层环境可能改变训练轨迹；短训练通过不能推出完整 100 轮重训会恢复同一论文分数。
