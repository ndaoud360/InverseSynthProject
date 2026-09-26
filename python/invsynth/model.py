"""
Phase 3 / Stage 1 - the Predictor.

Variable-length handling
------------------------
* All layers before pooling are convolutional, so any time length T works.
* Batches are padded with the "silence" feature value (-1) and a boolean mask
  of valid frames is carried through every time-downsampling step.
* A time-coordinate channel (0 at the first valid frame, 1 at the last) is
  appended to the input so the network can still reason about *where* an
  event happens (attack vs. release) after the global pool, independent of
  the clip length (CoordConv).
* Masked Attentive Statistics Pooling produces a fixed 2C vector (weighted
  mean + std over time) regardless of T, ignoring padded frames.
* At export time batch=1 and no padding exists, so the mask is all-true and
  the ONNX graph simply has a dynamic time axis.

  mel (B,128,T) + coord ─> ResNet2D (4 stages, /8 time, /16 freq)
      ─> reshape (B, C*F, T/8) ─> 1x1 Conv ─> 2 dilated TCN blocks
      ─> masked attentive stats pooling (B, 2C)
      ─> concat ctx-embedding ─> MLP ─┬─> 36 continuous (sigmoid)
                                      └─> 5 categorical heads (softmax)
"""
import torch
import torch.nn as nn
import torch.nn.functional as Fnn

from .params import CAT_CLASSES, CAT_IDX, CONT_IDX, N_PARAMS

PAD_VALUE = -1.0


class ResBlock2d(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.sc = None
        if stride != (1, 1) or cin != cout:
            self.sc = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False),
                                    nn.BatchNorm2d(cout))

    def forward(self, x):
        y = Fnn.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return Fnn.relu(y + (x if self.sc is None else self.sc(x)))


class TCNBlock(nn.Module):
    def __init__(self, c, dilation):
        super().__init__()
        self.conv = nn.Conv1d(c, c, 3, padding=dilation, dilation=dilation)
        self.bn = nn.BatchNorm1d(c)

    def forward(self, x, mask):
        y = Fnn.relu(self.bn(self.conv(x * mask)))
        return x + y


class AttentiveStatsPool(nn.Module):
    def __init__(self, c, hidden=128):
        super().__init__()
        self.att = nn.Sequential(nn.Conv1d(c, hidden, 1), nn.Tanh(), nn.Conv1d(hidden, c, 1))

    def forward(self, h, mask):                 # h (B,C,T), mask (B,1,T) float
        logits = self.att(h)
        logits = logits.masked_fill(mask < 0.5, -1e4)
        a = torch.softmax(logits, dim=-1)
        mu = (a * h).sum(-1)
        var = (a * h * h).sum(-1) - mu * mu
        return torch.cat([mu, torch.sqrt(var.clamp(min=1e-5))], dim=1)


def _down_len(lengths, n):
    for _ in range(n):
        lengths = torch.div(lengths - 1, 2, rounding_mode="floor") + 1
    return lengths


class InverseSynthNet(nn.Module):
    def __init__(self, n_mels=128, width=(32, 64, 128, 192, 256), emb=384):
        super().__init__()
        w = width
        self.stem = nn.Sequential(nn.Conv2d(2, w[0], 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(w[0]), nn.ReLU())
        strides = [(2, 2), (2, 2), (2, 2), (2, 1)]      # (freq, time)
        self.t_down = sum(1 for s in strides if s[1] == 2)
        layers = []
        for i, s in enumerate(strides):
            layers += [ResBlock2d(w[i], w[i + 1], s), ResBlock2d(w[i + 1], w[i + 1], (1, 1))]
        self.resnet = nn.Sequential(*layers)
        f_out = n_mels // 16
        self.proj = nn.Sequential(nn.Conv1d(w[-1] * f_out, emb, 1), nn.BatchNorm1d(emb), nn.ReLU())
        self.tcn = nn.ModuleList([TCNBlock(emb, 1), TCNBlock(emb, 2), TCNBlock(emb, 4)])
        self.pool = AttentiveStatsPool(emb)
        self.ctx = nn.Sequential(nn.Linear(3, 32), nn.ReLU())
        self.mlp = nn.Sequential(nn.Linear(2 * emb + 32, 512), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(512, 512), nn.ReLU())
        self.head_cont = nn.Linear(512, len(CONT_IDX))
        self.head_cat = nn.ModuleList([nn.Linear(512, c) for c in CAT_CLASSES])

    def forward(self, mel, ctx, lengths=None):
        """
        mel: (B, 128, T) features, padded with PAD_VALUE.  ctx: (B, 3).
        lengths: (B,) valid frames, or None (= all frames valid, used for ONNX).
        Returns (continuous (B,36) in [0,1], list of categorical logits).
        """
        B, M, T = mel.shape
        t = torch.arange(T, device=mel.device, dtype=mel.dtype).unsqueeze(0)
        if lengths is None:
            lengths_f = torch.full((B, 1), 1.0, device=mel.device, dtype=mel.dtype) * T
        else:
            lengths_f = lengths.to(mel.dtype).unsqueeze(1)
        coord = (t / (lengths_f - 1.0).clamp(min=1.0)).clamp(max=1.0)       # (B,T)
        valid = (t < lengths_f).to(mel.dtype)
        coord = coord * valid
        x = torch.stack([mel, coord.unsqueeze(1).expand(B, M, T)], dim=1)  # (B,2,M,T)

        h = self.resnet(self.stem(x))                                      # (B,C,F,T')
        h = h.flatten(1, 2)                                                # (B,C*F,T')
        Tp = h.shape[-1]
        if lengths is None:
            mask = torch.ones(B, 1, Tp, device=h.device, dtype=h.dtype)
        else:
            lp = _down_len(lengths, self.t_down).clamp(min=1)
            mask = (torch.arange(Tp, device=h.device)[None, :] < lp[:, None]).to(h.dtype)
            mask = mask.unsqueeze(1)
        h = self.proj(h) * mask
        for blk in self.tcn:
            h = blk(h, mask) * mask
        z = torch.cat([self.pool(h, mask), self.ctx(ctx)], dim=1)
        z = self.mlp(z)
        return torch.sigmoid(self.head_cont(z)), [hd(z) for hd in self.head_cat]


def assemble_vector(cont, cat_logits):
    """(B,36) + logits -> (B,41) normalised parameter vector (bin centres for cats)."""
    B = cont.shape[0]
    out = cont.new_zeros(B, N_PARAMS)
    out[:, CONT_IDX] = cont
    for i, (idx, n) in enumerate(zip(CAT_IDX, CAT_CLASSES)):
        k = cat_logits[i].argmax(dim=1).to(cont.dtype)
        out[:, idx] = (k + 0.5) / n
    return out


class ExportWrapper(nn.Module):
    """Single-tensor output for ONNX: params (1,41)."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, mel, ctx):
        cont, cats = self.net(mel, ctx, None)
        return assemble_vector(cont, cats)


def param_loss(cont, cat_logits, y, rel, cat_weight=0.3):
    """
    Relevance-masked loss. y, rel: (B,41). Irrelevant parameters (see
    params.relevance_mask) contribute nothing, which removes a large source of
    label noise (e.g. reverb size when reverb mix is 0).
    """
    yc, rc = y[:, CONT_IDX], rel[:, CONT_IDX]
    l_cont = ((cont - yc) ** 2 * rc).sum() / rc.sum().clamp(min=1.0)
    l_cat = cont.new_zeros(())
    for i, (idx, n) in enumerate(zip(CAT_IDX, CAT_CLASSES)):
        tgt = (y[:, idx] * n).long().clamp(0, n - 1)
        ce = Fnn.cross_entropy(cat_logits[i], tgt, reduction="none")
        l_cat = l_cat + (ce * rel[:, idx]).sum() / rel[:, idx].sum().clamp(min=1.0)
    return l_cont + cat_weight * l_cat / len(CAT_IDX), l_cont.detach(), l_cat.detach()
