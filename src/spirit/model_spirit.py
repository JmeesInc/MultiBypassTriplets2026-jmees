"""SPIRIT (arXiv:2608.02188) 再実装 — 段階的。

Stage A: Backbone(§4.1) + TUF(§4.2, text-conditioned unary features) + unary heads + 直接ivt head。
後で PIC(§4.3/4.4) と TGR(§4.5) を足す。設計/式番号は workspace/expA04_spirit/SESSION_NOTES.md 参照。

入力は「凍結 DINOv3-L の per-frame 8x8 pooled 特徴」を T=8 スタックしたもの: feats [B, T, N, D_bb]
  (D_bb=1024, N=64=8x8)。backbone は学習ループ外でキャッシュ済み（まず完全凍結で head を検証）。

deviation（SESSION_NOTES記載）:
- text encoder は当面 learnable class-query embedding で代替（意味初期化は視覚attnで上書きされる）。
  config text_init で将来 CLIP 埋め込みに差し替え可能。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

NI, NV, NT, NIVT = 12, 13, 15, 85


def sinusoidal_pe(n, d, device=None):
    """[n, d] 正弦波位置埋め込み。"""
    pe = torch.zeros(n, d, device=device)
    pos = torch.arange(n, dtype=torch.float32, device=device).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2, dtype=torch.float32, device=device) * (-math.log(10000.0) / d))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class TUFBranch(nn.Module):
    """Text-conditioned Unary Features の 1 枝（§4.2, eq6-9）。

    初期 query(nκ×d) を CA(視覚memory)→SA(枝内)→FFN→TCA(時間descriptor) で精緻化 → Q^κ(nκ×d)。
    """

    def __init__(self, ncls, d=128, nhead=4, ffn_mult=2.0, dropout=0.05):
        super().__init__()
        self.ncls = ncls
        self.query = nn.Embedding(ncls, d)  # text-init 代替（learnable class query）
        nn.init.normal_(self.query.weight, std=0.02)
        # CA: query <- visual memory M (eq6)
        self.ca = nn.MultiheadAttention(d, nhead, dropout=dropout, batch_first=True)
        self.ln_ca = nn.LayerNorm(d)
        # SA: 枝内 self-attn (eq7)
        self.sa = nn.MultiheadAttention(d, nhead, dropout=dropout, batch_first=True)
        self.ln_sa = nn.LayerNorm(d)
        # FFN (eq8)
        h = int(d * ffn_mult)
        self.ffn = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Dropout(dropout), nn.Linear(h, d))
        self.ln_ffn = nn.LayerNorm(d)
        # TCA: 時間 cross-attn, query <- frame descriptor G (eq9)
        self.tca = nn.MultiheadAttention(d, nhead, dropout=dropout, batch_first=True)
        self.ln_tca = nn.LayerNorm(d)

    def forward(self, M, G):
        """M: [B, T*N, d] 視覚memory / G: [B, T, d] frame descriptor -> Q^κ: [B, ncls, d]."""
        B = M.shape[0]
        q = self.query.weight.unsqueeze(0).expand(B, -1, -1)          # [B, ncls, d]
        # CA
        u, _ = self.ca(q, M, M)
        qbar = self.ln_ca(q + u)                                       # eq6
        # SA
        s, _ = self.sa(qbar, qbar, qbar)
        qtil = self.ln_sa(qbar + s)                                    # eq7
        # FFN
        qstar = self.ln_ffn(qtil + self.ffn(qtil))                    # eq8
        # TCA (時間)
        t, _ = self.tca(qstar, G, G)
        qk = self.ln_tca(qstar + t)                                   # eq9
        return qk


class SpiritStageA(nn.Module):
    """Stage A: Backbone射影 + PE + 3×TUF + unary heads + 直接ivt head（pairwise/graph なし）。

    ivt は pairwise 無しなので scene descriptor から直接 triplet head で予測（ablation baseline 相当）。
    """

    def __init__(self, d_bb=1024, d=128, ds=256, nhead=4, dropout=0.05, T=8, N=64):
        super().__init__()
        self.d, self.T, self.N = d, T, N
        self.vis_proj = nn.Linear(d_bb, d)                            # Πvision (eq2)
        self.register_buffer("spe", sinusoidal_pe(N, d), persistent=False)   # spatial PE
        self.register_buffer("tpe", sinusoidal_pe(T, d), persistent=False)   # temporal PE
        self.tuf_i = TUFBranch(NI, d, nhead, dropout=dropout)
        self.tuf_v = TUFBranch(NV, d, nhead, dropout=dropout)
        self.tuf_t = TUFBranch(NT, d, nhead, dropout=dropout)
        self.head_i = nn.Linear(d, 1)                                # fκ: 各classクエリ -> scalar logit
        self.head_v = nn.Linear(d, 1)
        self.head_t = nn.Linear(d, 1)
        # scene descriptor s (§4.3) + 直接 ivt head（Stage A 用）
        self.scene = nn.Sequential(nn.Linear(d, ds), nn.GELU())
        self.head_ivt = nn.Linear(ds, NIVT)

    def encode(self, feats):
        """feats [B,T,N,d_bb] -> Z [B,T,N,d], M [B,T*N,d], G [B,T,d], scene s [B,ds]."""
        B, T, N, _ = feats.shape
        Z0 = self.vis_proj(feats)                                    # [B,T,N,d] (eq2)
        Z = Z0 + self.spe.view(1, 1, N, self.d) + self.tpe.view(1, T, 1, self.d)  # eq3
        M = Z.reshape(B, T * N, self.d)                              # visual memory
        G = Z.mean(dim=2)                                            # frame descriptor [B,T,d]
        s = self.scene(M.mean(dim=1))                               # scene descriptor [B,ds]
        return Z, M, G, s

    def forward(self, feats):
        _, M, G, s = self.encode(feats)
        Qi, Qv, Qt = self.tuf_i(M, G), self.tuf_v(M, G), self.tuf_t(M, G)  # [B,ncls,d]
        oi = self.head_i(Qi).squeeze(-1)                            # [B,NI]
        ov = self.head_v(Qv).squeeze(-1)                            # [B,NV]
        ot = self.head_t(Qt).squeeze(-1)                            # [B,NT]
        oivt = self.head_ivt(s)                                     # [B,85]（Stage A: 直接）
        return {"ivt": oivt, "i": oi, "v": ov, "t": ot,
                "Qi": Qi, "Qv": Qv, "Qt": Qt, "scene": s}           # Q* は次段 PIC で使う


if __name__ == "__main__":  # CPU forward smoke
    torch.manual_seed(0)
    B, T, N, Dbb = 2, 8, 64, 1024
    m = SpiritStageA(d_bb=Dbb, T=T, N=N).eval()
    feats = torch.randn(B, T, N, Dbb)
    with torch.no_grad():
        o = m(feats)
    for k in ("ivt", "i", "v", "t"):
        print(f"{k}: {tuple(o[k].shape)}")
    assert o["ivt"].shape == (B, NIVT) and o["i"].shape == (B, NI)
    assert o["v"].shape == (B, NV) and o["t"].shape == (B, NT)
    assert o["Qi"].shape == (B, NI, 128) and o["Qt"].shape == (B, NT, 128)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"params: {n_params/1e6:.2f}M")
    print("STAGE A SMOKE OK")
