# 相对官方 Medformer 的变动

上游：`891f65b8a7e77188508fd8c56e10ccdba7fc1e18`。49 个官方 Python 文件逐字节保留。run.py、官方数据工厂、Exp_Classification、模型、层和工具文件均不编辑。

替换根 README 为 PDSigformer 说明，原文保留为 README_MEDFORMER.md；扩展 .gitignore 排除本地产物。根 LICENSE 保留上游 MIT 条款与署名，并加入远端项目已有的 2026 SI-ML 署名。

新增 .gitattributes 保留来源文件的精确字节，防止 Windows 自动换行转换破坏 SHA 校验。

移除未被本文实验调用的 7 个上游 Notebook：`experiments.ipynb`、`supplement_experiments.ipynb` 和 `data_preprocessing/` 中的 5 个原数据集预处理 Notebook。原始 SHA 清单保留，源码核验单独记录这些已批准的删减。PADS/Shimmer 预处理目录暂时提供两个空白 Python 文件及流程说明，尚未实现原始数据处理；本文训练和推理仍加载冻结的已处理数据。

| 新增文件 / 目录 | 原因与影响 |
|---|---|
| models/PDSigformer.py、pds_components.py、pds_timesnet.py | 已保存实验的残差模型与前端，不改变官方 Medformer |
| models/TimesNet.py、models/TeCh.py | 独立基线及接口；TeCh 保留 core. 权重前缀 |
| third_party/medformer_paper/ | 历史权重需要旧结构，只迁移导入命名空间 |
| third_party/timesnet/ | 固定基线依赖，保留来源和许可证 |
| data_provider/wearable_loader.py | 加载固定缓存，复用官方 collate，保持数组与 mask |
| exp/exp_wearable.py | 继承官方分类类，独立训练循环，训练不读取 Test |
| utils/pds_checkpoint.py、pds_runtime.py | SHA、指标、随机流、非张量状态和历史运行库 |
| run_wearable.py、configs/paper.json | 明确 seed，避开原入口 itr 的自动设种子行为 |
| scripts/、reproduction/、docs/ | 调用、排队、验收、清单和报告 |

Reformer 兼容器原来按旧模块名称绑定 GELU；迁移后对隔离命名空间中的同一类做严格类型、层数与激活检查。公式、PTX 和模型参数不改。

新增 loader 是为了固定 PADS/Shimmer 缓存及其身份检查，不需要修改官方 data_factory.py。原官方入口仍可加载已处理好的原数据集；如需原始数据预处理 Notebook，可从上游 Medformer 仓库获取。

`python scripts/audit_source.py` 检查上游文件；若新克隆的仓库未包含原 Medformer 提交对象，则使用随包提供的原始 SHA 清单核验，不要求重新拉取上游历史。仅对上述 7 个 Notebook 的缺失作明确记录，其余文件缺失仍会失败。此兼容处理只影响源码核验，不参与训练或推理。既有发布测试范围见 reproduction/RELEASE_VALIDATION.md。
