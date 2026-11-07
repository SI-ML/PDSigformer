这里保留论文实验使用的固定依赖版本，防止覆盖主仓的官方 Medformer main 文件。

| 来源 | 固定版本 | 用途 |
|---|---|---|
| DL4mHealth/Medformer | `446275f27b713a9f09917a6ba0bc51a18e921597` | 历史基线的模型与层；只调整 Python 导入命名空间 |
| thuml/Time-Series-Library | `4e938a1767106324dd753b2a44832bf870a0252e` | TimesNet 基线及依赖 |
| Levi-Ackman/TeCh | `9a378cc546a5d97c871eff282148175b3c7cd75b` | 可选基线，使用下载脚本安装 |

主仓 `models/Medformer.py` 仍为 `891f65b8a7e77188508fd8c56e10ccdba7fc1e18`，PDSigformer 后端使用此版本。历史 Medformer 基线使用旧版及其原有 SWA 配置，两者用途不同，不能交换 checkpoint。

来源和逐文件摘要见 `SOURCES.json`、`VENDOR_MANIFEST.json`。上游许可文件分别保留在对应目录；根目录 LICENSE 保留官方 Medformer 的署名。
