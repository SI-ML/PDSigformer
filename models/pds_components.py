"""TimesNet 时序特征串联原 Medformer 的固定长度分类扩展。"""

from copy import deepcopy

import torch
from torch import nn

from layers.Embed import DataEmbedding
from models.Medformer import Model as MedformerModel
from models.pds_timesnet import TimesBlock


class TimesNetFeatureExtractor(nn.Module):
    """复用官方分类分支到最后一个 LayerNorm，保留时间轴。"""

    def __init__(self, configs):
        super().__init__()
        front = deepcopy(configs)
        front.d_model = getattr(configs, "ts_d_model", 32)
        front.d_ff = getattr(configs, "ts_d_ff", 64)
        front.e_layers = getattr(configs, "ts_e_layers", 2)
        front.top_k = getattr(configs, "ts_top_k", 3)
        front.num_kernels = getattr(configs, "ts_num_kernels", 3)

        if front.seq_len < 2 or front.seq_len > 5000:
            raise ValueError("PDSigformer 要求 seq_len 在 2 到 5000 的范围内。")
        # 下游 ListPatchEmbedding 将 seq_len 用作通道位置编码的 d_model。
        if front.seq_len % 2:
            raise ValueError("原 Medformer 通道位置编码使用 d_model=seq_len，要求 seq_len 为偶数。")
        if front.d_model <= 0 or front.d_model % 2:
            raise ValueError("ts_d_model 必须为正偶数，以兼容官方位置编码。")
        if front.d_ff <= 0 or front.e_layers <= 0 or front.num_kernels <= 0:
            raise ValueError("ts_d_ff、ts_e_layers、ts_num_kernels 必须为正整数。")
        if not 1 <= front.top_k <= front.seq_len // 2:
            raise ValueError("ts_top_k 必须在 1 到 seq_len // 2 的范围内。")
        if front.pred_len != 0:
            raise ValueError("PDSigformer 分类要求 pred_len=0。")

        self.seq_len = front.seq_len
        self.enc_in = front.enc_in
        self.output_dim = front.d_model
        self.enc_embedding = DataEmbedding(
            front.enc_in, front.d_model, front.embed, front.freq, front.dropout
        )
        self.blocks = nn.ModuleList(
            [TimesBlock(front) for _ in range(front.e_layers)]
        )
        # 与官方 TimesNet 一致：每个 block 后复用同一个 LayerNorm。
        self.layer_norm = nn.LayerNorm(front.d_model)

    def forward(self, x):
        hidden = self.enc_embedding(x, None)
        for block in self.blocks:
            hidden = self.layer_norm(block(hidden))
        return hidden


class Model(nn.Module):
    """[B,T,C] → TimesNet 隐特征 [B,T,D_ts] → Medformer → [B,K]。"""

    def __init__(self, configs):
        super().__init__()
        if configs.task_name != "classification":
            raise ValueError("PDSigformer 当前仅支持 classification。")
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.frontend = TimesNetFeatureExtractor(configs)
        backend = deepcopy(configs)
        # 潜在特征作为下游输入通道；不改变共享 args 或原 Medformer 实现。
        backend.enc_in = self.frontend.output_dim
        self.backbone = MedformerModel(backend)

    def _validate_input(self, x_enc, padding_mask):
        if x_enc.ndim != 3 or tuple(x_enc.shape[1:]) != (
            self.seq_len, self.enc_in
        ):
            raise ValueError(
                f"PDSigformer 输入必须为 [B,{self.seq_len},{self.enc_in}]。"
            )
        if x_enc.shape[0] == 0 or not x_enc.is_floating_point():
            raise ValueError("PDSigformer 需要非空的浮点输入 batch。")
        if not bool(torch.isfinite(x_enc).all()):
            raise ValueError("PDSigformer 输入含 NaN 或 Inf。")
        if padding_mask is None or tuple(padding_mask.shape) != tuple(x_enc.shape[:2]):
            raise ValueError("padding mask 必须提供，形状为 [B,T]。")
        if not bool((padding_mask == 1).all()):
            raise ValueError(
                "PDSigformer 仅支持固定长度、全有效窗口；padding mask 必须全部为 1。"
            )

    def classification(self, x_enc, x_mark_enc):
        self._validate_input(x_enc, x_mark_enc)
        # 全有效 mask 已检查，前后无需补零；不混入分类 logits 或切断梯度。
        hidden = self.frontend(x_enc)
        return self.backbone(hidden, x_mark_enc, None, None)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if mask is not None:
            raise ValueError("PDSigformer 不支持 imputation mask；请使用第二参数传 padding mask。")
        return self.classification(x_enc, x_mark_enc)
