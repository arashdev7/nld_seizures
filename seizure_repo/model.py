"""
DC-SA-EBiLSTM (Wang et al., 2026) - built from Table 2, Figures 3-4, Eqs. 5-10.

Input: (B, 432) feature vector -> reshaped to (B, 18 channels, 24 features).
Important: every "sequence" operation here (conv, attention, LSTM) runs over the
18-CHANNEL axis, not over time. Each 1-s window is classified independently.

Ablation flags reproduce the variants in Table 4.
"""
import torch
import torch.nn as nn


class DSConvBlock(nn.Module):
    """Depthwise (k=3, groups=64) + pointwise conv over the channel axis, BN+ReLU,
    dropout, residual (Eq. 6-7, Figure 3)."""
    def __init__(self, d=64, k=3, dropout=0.2, depthwise=True):
        super().__init__()
        if depthwise:
            self.conv = nn.Sequential(
                nn.Conv1d(d, d, k, padding=k // 2, groups=d),   # DWConv
                nn.Conv1d(d, d, 1),                              # PWConv
            )
        else:                                                    # ablation: standard Conv1d
            self.conv = nn.Conv1d(d, d, k, padding=k // 2)
        self.bn = nn.BatchNorm1d(d)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, h):                    # h: (B, C, d)
        z = h.transpose(1, 2)                # (B, d, C): conv slides across EEG channels
        z = self.act(self.bn(self.conv(z))).transpose(1, 2)
        return h + self.drop(z)              # Eq. 7


class EBiLSTM(nn.Module):
    """BiLSTM + residual projection + gated fusion + LayerNorm (Eq. 9-10, Figure 4)."""
    def __init__(self, d_in=64, hidden=32, gate=True, residual=True, layernorm=True,
                 enhanced=True, bidirectional=True):
        super().__init__()
        if not bidirectional:                # plain LSTM baseline: same output width
            hidden *= 2
        self.lstm = nn.LSTM(d_in, hidden, num_layers=1, batch_first=True,
                            bidirectional=bidirectional)
        out = hidden * (2 if bidirectional else 1)
        self.enhanced, self.gate_on, self.res_on = enhanced, gate, residual
        self.proj = nn.Linear(d_in, out)                 # "Linear(C_in -> 2H)"
        self.gate = nn.Linear(2 * out, out)              # "Linear(4H -> 2H)"
        self.ln = nn.LayerNorm(out) if layernorm else nn.Identity()

    def forward(self, x):                    # x: (B, C, d)
        h, _ = self.lstm(x)
        if not self.enhanced:                # ablation: plain BiLSTM
            return h
        if not self.res_on:                  # ablation: no residual projection branch
            return self.ln(h)
        p = self.proj(x)
        if self.gate_on:
            g = torch.sigmoid(self.gate(torch.cat([h, p], dim=-1)))   # Eq. 9
            return self.ln(g * h + (1 - g) * p)                       # Eq. 10
        return self.ln(h + p)                # ablation: no gate -> plain residual sum


class DCSAEBiLSTM(nn.Module):
    def __init__(self, n_ch=18, n_feat=24, d=64, heads=4, ffn_ratio=2, dropout=0.2,
                 lstm_hidden=32, fc_hidden=128, fc_dropout=0.3, n_classes=2,
                 use_dsconv=True, use_conv=True, use_attention=True, use_posenc=True,
                 use_lstm=True, ebilstm_kwargs=None):
        super().__init__()
        self.n_ch, self.n_feat = n_ch, n_feat
        self.embed = nn.Linear(n_feat, d)                                   # Eq. 5
        self.local = DSConvBlock(d, 3, dropout, depthwise=use_dsconv) if use_conv else nn.Identity()
        self.use_posenc = use_posenc
        self.pos = nn.Parameter(torch.zeros(1, n_ch, d))                    # learnable PE
        nn.init.trunc_normal_(self.pos, std=0.02)
        # Attention + residual + FFN (Section 2.4.2). Post-norm is PyTorch's default;
        # the paper doesn't specify norm placement.
        self.attn = (nn.TransformerEncoderLayer(d, heads, dim_feedforward=ffn_ratio * d,
                                                dropout=dropout, batch_first=True)
                     if use_attention else nn.Identity())
        self.ebilstm = (EBiLSTM(d, lstm_hidden, **(ebilstm_kwargs or {}))
                        if use_lstm else nn.Identity())      # identity keeps 18 x 64
        self.head = nn.Sequential(                                          # Section 2.4.4
            nn.Flatten(),
            nn.Linear(n_ch * 2 * lstm_hidden, fc_hidden),
            nn.BatchNorm1d(fc_hidden),
            nn.ReLU(),
            nn.Dropout(fc_dropout),
            nn.Linear(fc_hidden, n_classes),
        )

    def forward(self, x):                    # x: (B, 432) or (B, 18, 24)
        x = x.reshape(-1, self.n_ch, self.n_feat)
        h = self.local(self.embed(x))
        if self.use_posenc:
            h = h + self.pos
        h = self.attn(h)
        h = self.ebilstm(h)
        return self.head(h)


# Table 4 variants
ABLATIONS = {
    "full":               dict(),
    "DC-SA-BiLSTM":       dict(ebilstm_kwargs=dict(enhanced=False)),
    "no_gate":            dict(ebilstm_kwargs=dict(gate=False)),
    "no_residual_proj":   dict(ebilstm_kwargs=dict(residual=False)),
    "no_layernorm":       dict(ebilstm_kwargs=dict(layernorm=False)),
    "no_self_attention":  dict(use_attention=False),
    "standard_conv1d":    dict(use_dsconv=False),
    "no_posenc":          dict(use_posenc=False),
    # Table 3 neural hybrids
    "SA-EBiLSTM":         dict(use_conv=False),
    "CNN-EBiLSTM":        dict(use_attention=False, use_posenc=False),
    # Table 3 plain neural baselines (same embedding + head, one component each)
    "CNN":                dict(use_attention=False, use_posenc=False, use_lstm=False),
    "LSTM":               dict(use_conv=False, use_attention=False, use_posenc=False,
                               ebilstm_kwargs=dict(enhanced=False, bidirectional=False)),
    "BiLSTM":             dict(use_conv=False, use_attention=False, use_posenc=False,
                               ebilstm_kwargs=dict(enhanced=False)),
}


if __name__ == "__main__":
    for name, kw in ABLATIONS.items():
        m = DCSAEBiLSTM(**kw)
        y = m(torch.randn(8, 432))
        n = sum(p.numel() for p in m.parameters())
        print(f"{name:20s} out={tuple(y.shape)} params={n:,}")
