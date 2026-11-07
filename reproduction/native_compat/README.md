# 历史数值兼容

三个 Python 模块逐字节保留原统一推理验收使用的源码。新入口使用这些公开文件，并将 SHA 与原 manifest 对照。运行库二进制从用户的 artifacts 导入，不随源码分发。

- TimesNet / Autoformer：原 cuFFT 10.9 路径，仅限锁定 Test batch 与形状。
- Reformer：CUDA 11.8 编译的 GELU。保留 kernel 源码、PTX 与上游 PyTorch 许可；当前验收 GPU 为 SM89。
- 此兼容层不参与训练反传，也不使用历史概率替换模型输出。

Reformer 类绑定的命名空间适配位于 utils/pds_runtime.py，原算子模块未修改。
