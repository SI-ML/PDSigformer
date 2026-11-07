# 发布验收记录

- 102 份唯一历史权重、108 个表格引用：全部对应。
- 204 个有效推理任务：每份权重两个独立进程；逐样本身份、标签、预测、概率与六指标通过。
- 六指标与历史记录误差均不超过 1e-12；概率按 atol=1e-6、rtol=1e-5 比较。
- 34 个配置均在 seed42 完成一个完整 epoch，验证有限梯度、参数更新、Valid 选模、保存和全新模型重载。
- **未对全部 102 组重新进行 100 epoch 训练，因此 fresh_full_training_verified=false。**
- 失败尝试保留：推理 14 次、训练 6 次；补测覆盖对应任务，没有覆盖原日志。
- 此轮失败来自记录器把 PyTorch 虚拟模块名当成文件；修复后补测通过。早期 Reformer 迁移绑定失败单独保留在 pilot 记录。
- 官方 Python 完整性由 SOURCE_AUDIT.json 单独核验。

## 可读结果

[离线 HTML 表格](PDSigformer_RESULTS.html)、[每 seed 指标](TEST_PER_SEED.csv)、[表格均值和样本标准差](PAPER_TABLE_MEAN_STD.csv)、[训练测试](TRAINING_CHECK.csv)。

旧权重通过推理验收与从头训练的软件链路通过是两种不同证据。此处不宣称随机训练的逐位轨迹或新训练分数等于历史结果。


## 干净源码包测试

源码压缩包在独立目录解压，TeCh 通过固定上游提交重新下载并验证 SHA。PDSigformer、TimesNet、TeCh 三个模型均重新完成一轮训练，新 checkpoint 随后通过独立 eval 命令推理；另有五个配置的历史权重在该目录推理通过，覆盖 FFT 与 GELU 兼容路径。

使用现有已验证 Python 环境，没有把这项测试描述为从空环境重新安装依赖。最终新增代码格式化后，103 个 Python 文件与被测源码包的 AST 全部一致；原官方 49 个 Python 文件仍逐字节一致。详见 PACKAGE_VALIDATION.json 和 PACKAGE_SOURCE_EQUIVALENCE.json。

两个数据集 Train/Valid/Test 受试者交集均为零，见 DATA_PROTOCOL_CHECK.json。
