TeCh 的固定来源是 https://github.com/Levi-Ackman/TeCh ，提交 `9a378cc546a5d97c871eff282148175b3c7cd75b`。

该固定版本未发现 LICENSE/COPYING 文件。发布压缩包不捆绑它的三个实现文件，也不为其声明 MIT 授权。需要此基线时运行 `python scripts/fetch_tech.py`，从上游获取并按 `third_party/VENDOR_MANIFEST.json` 验证原始文件与导入适配后的 SHA256。请遵守上游的使用条件。

`models/TeCh.py` 仅负责把 Medformer 四参数接口转发到 TeCh 的 `forward(x)`。
