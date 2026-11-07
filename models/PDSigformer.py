"""PDSigformer with serial, channel-projected, or input-residual connections."""

from copy import deepcopy

from torch import nn
from torch.nn import functional as F

from models.Medformer import Model as MedformerModel
from models.pds_components import Model as LegacyPDSigformer
from models.pds_components import TimesNetFeatureExtractor


class Model(nn.Module):
    """Reuse the frontend and backbone while selecting the connection between them."""

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

        # Keep module names and initialization order identical for same-seed projected/residual comparisons.
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
        # Reuse input validation without constructing the legacy model or its unused backbone.
        LegacyPDSigformer._validate_input(self, x_enc, padding_mask)

    def classification(self, x_enc, x_mark_enc):
        self._validate_input(x_enc, x_mark_enc)
        hidden = self.bridge_dropout(F.gelu(self.frontend(x_enc)))
        hidden = hidden * x_mark_enc.unsqueeze(-1).to(
            device=hidden.device, dtype=hidden.dtype
        )
        backend_input = self.projection(hidden)
        if self.variant == "residual":
            # X is the preprocessed tensor delivered by the loader, not the raw signal on disk.
            backend_input = x_enc + backend_input
        return self.backbone(backend_input, x_mark_enc, None, None)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if mask is not None:
            raise ValueError("PDSigformerPilot 不支持 imputation mask；第二参数为 padding mask。")
        return self.classification(x_enc, x_mark_enc)
