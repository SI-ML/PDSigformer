"""比较串联、通道投影和原始输入残差的独立 PDS 试验模型。"""

from copy import deepcopy

from torch import nn
from torch.nn import functional as F

from models.Medformer import Model as MedformerModel
from models.pds_components import Model as LegacyPDSigformer
from models.pds_components import TimesNetFeatureExtractor


class Model(nn.Module):
    """复用既有前后端，只在两者之间显式区分三种连接方式。"""

    def __init__(self, configs):
        super().__init__()
        if configs.task_name != "classification":
            raise ValueError("PDSigformerPilot 当前仅支持 classification。")
        self.variant = getattr(configs, "pds_variant", "residual")
        if self.variant not in ("serial", "projected", "residual"):
            raise ValueError("pds_variant 必须为 serial、projected 或 residual。")
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.frontend = TimesNetFeatureExtractor(configs)
        self.bridge_dropout = nn.Dropout(configs.dropout)

        # projected/residual 的模块名称与初始化顺序一致，便于同种子对照。
        self.projection = (
            nn.Identity()
            if self.variant == "serial"
            else nn.Linear(self.frontend.output_dim, self.enc_in, bias=True)
        )
        backend = deepcopy(configs)
        backend.enc_in = (
            self.frontend.output_dim if self.variant == "serial" else self.enc_in
        )
        self.backbone = MedformerModel(backend)

    def _validate_input(self, x_enc, padding_mask):
        # 仅复用输入检查，不构造旧模型及其无用的 backbone。
        LegacyPDSigformer._validate_input(self, x_enc, padding_mask)

    def classification(self, x_enc, x_mark_enc):
        self._validate_input(x_enc, x_mark_enc)
        hidden = self.bridge_dropout(F.gelu(self.frontend(x_enc)))
        hidden = hidden * x_mark_enc.unsqueeze(-1).to(
            device=hidden.device, dtype=hidden.dtype
        )
        backend_input = self.projection(hidden)
        if self.variant == "residual":
            # 这里的 X 是 loader 实际交给模型的输入，不是未预处理的磁盘信号。
            backend_input = x_enc + backend_input
        return self.backbone(backend_input, x_mark_enc, None, None)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if mask is not None:
            raise ValueError("PDSigformerPilot 不支持 imputation mask；第二参数为 padding mask。")
        return self.classification(x_enc, x_mark_enc)
