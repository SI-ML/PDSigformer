"""TeCh 的 Medformer 接口；保留历史 checkpoint 的 core. 前缀。"""
from torch import nn


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        try:
            from third_party.tech.model import Model as TeChModel
        except ModuleNotFoundError as exc:
            raise ImportError("先运行 python scripts/fetch_tech.py 安装固定版本 TeCh") from exc
        self.core = TeChModel(configs)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        return self.core(x_enc)
