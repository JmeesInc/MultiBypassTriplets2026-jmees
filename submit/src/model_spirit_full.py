"""SPIRIT 非蒸留フル版: TUF(§4.2) + PIC(§4.3/4.4 pairwise I-T/I-V) + TGR(§4.5 triplet graph)。

Stage A(TUF) の上に pairwise 関係推論と三重項合成グラフを積む。ivt は TGR から出力。
蒸留(§4.6, Stage E)は別途。設計/式番号は SESSION_NOTES.md 参照。

微小グラフ(ni+nt=27ノード, ni*nt=180エッジ)なので torch_geometric 不使用の dense 実装。
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_spirit import TUFBranch, sinusoidal_pe, NI, NV, NT, NIVT

NIT, NIV = NI * NT, NI * NV   # 180, 156 (dense grid)

# triplet k -> dense it/iv relation-node index（component_maps から, cache 済み）
_HERE = os.path.dirname(__file__)


def _load_maps():
    import sys
    sys.path.insert(0, os.path.join(_HERE, "..", "..", "..", "reference/starter_kit"))
    from utils.triplet_mappings import triplet_maps
    CM = np.array(triplet_maps["multibypasst40"]["component_maps"])  # [85,6] ivt,i,v,t,iv,it
    it_flat = (CM[:, 1] * NT + CM[:, 3]).astype(np.int64)   # [85] -> 0..179
    iv_flat = (CM[:, 1] * NV + CM[:, 2]).astype(np.int64)   # [85] -> 0..155
    return it_flat, iv_flat


class BipartiteGATv2(nn.Module):
    """dense 二部グラフ GATv2（edge attr 付き, multi-head）。A↔B 双方向 message passing。

    A[B,na,d], Bn[B,nb,d], E[B,na,nb,de] -> A'[B,na,do], B'[B,nb,do]。
    GATv2: score(m,n)=a^T LeakyReLU(Ws h + Wt h' + We e)（非線形を attn ベクトルの前に）。
    """

    def __init__(self, d, de, do, heads=4):
        super().__init__()
        self.h, self.dh = heads, do // heads
        self.Ws = nn.Linear(d, do); self.Wt = nn.Linear(d, do); self.We = nn.Linear(de, do)
        self.Va = nn.Linear(d, do); self.Vb = nn.Linear(d, do)
        self.a = nn.Parameter(torch.randn(heads, self.dh) * 0.02)
        self.leaky = nn.LeakyReLU(0.2)

    def _score(self, hs, ht, e):
        # hs[B,na,do] ht[B,nb,do] e[B,na,nb,do] -> score[B,h,na,nb]
        B, na, _ = hs.shape; nb = ht.shape[1]
        x = self.leaky(hs[:, :, None] + ht[:, None] + e)             # [B,na,nb,do]
        x = x.view(B, na, nb, self.h, self.dh)
        return torch.einsum("bmnhd,hd->bhmn", x, self.a)             # [B,h,na,nb]

    def forward(self, A, Bn, E):
        hs, ht, e = self.Ws(A), self.Wt(Bn), self.We(E)             # [B,na/nb,do], edge [B,na,nb,do]
        s = self._score(hs, ht, e)                                  # [B,h,na,nb]
        va = self.Va(A).view(*A.shape[:2], self.h, self.dh)         # [B,na,h,dh]
        vb = self.Vb(Bn).view(*Bn.shape[:2], self.h, self.dh)       # [B,nb,h,dh]
        # A gathers from B (softmax over n)
        aA = torch.softmax(s, dim=3)                                # [B,h,na,nb]
        A2 = torch.einsum("bhmn,bnhd->bmhd", aA, vb).reshape(*A.shape[:2], -1)
        # B gathers from A (softmax over m)
        aB = torch.softmax(s, dim=2)                                # [B,h,na,nb]
        B2 = torch.einsum("bhmn,bmhd->bnhd", aB, va).reshape(*Bn.shape[:2], -1)
        return A2, B2


class PICBranch(nn.Module):
    """Pairwise Interaction Coupling の 1 枝（I-T or I-V, §4.3/4.4）。

    Qa[B,na,d], Qb[B,nb,d], scene s[B,ds] -> 精緻化 node Q̃a,Q̃b, edge埋込 E[B,na,nb,de], pairwise logit[B,na,nb]。
    """

    def __init__(self, na, nb, d=128, ds=256, de=128, do=256, heads=4, dropout=0.05):
        super().__init__()
        self.na, self.nb = na, nb
        self.edge_map = nn.Sequential(nn.Linear(2 * d + ds, de), nn.GELU(), nn.Linear(de, de))  # ϕ (eq11)
        self.gat = BipartiteGATv2(d, de, do, heads)                # eq12
        self.res_a = nn.Linear(d, do); self.res_b = nn.Linear(d, do)  # Πres (eq13)
        self.ln_a = nn.LayerNorm(do); self.ln_b = nn.LayerNorm(do)
        self.psi_a = nn.Linear(de, do); self.psi_b = nn.Linear(de, do)  # edge集約 (eq14)
        self.ffn_a = nn.Sequential(nn.Linear(do, 2 * do), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * do, do))
        self.ffn_b = nn.Sequential(nn.Linear(do, 2 * do), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * do, do))
        self.ln_fa = nn.LayerNorm(do); self.ln_fb = nn.LayerNorm(do)
        self.edge_cls = nn.Linear(de, 1)                            # g (eq17)

    def forward(self, Qa, Qb, s):
        B = Qa.shape[0]
        # pair descriptor r=[qa; qb; s] -> edge embedding E (eq11)
        sa = s[:, None, None].expand(B, self.na, self.nb, s.shape[-1])
        r = torch.cat([Qa[:, :, None].expand(B, self.na, self.nb, Qa.shape[-1]),
                       Qb[:, None].expand(B, self.na, self.nb, Qb.shape[-1]), sa], dim=-1)
        E = self.edge_map(r)                                        # [B,na,nb,de]
        # GATv2 message passing + residual (eq12,13)
        A2, B2 = self.gat(Qa, Qb, E)
        Qa_g = self.ln_a(self.res_a(Qa) + A2)
        Qb_g = self.ln_b(self.res_b(Qb) + B2)
        # edge aggregation to nodes (eq14,15)
        aa = self.psi_a(E).mean(dim=2)                              # [B,na,do] over nb
        ab = self.psi_b(E).mean(dim=1)                              # [B,nb,do] over na
        Qa_h = Qa_g + aa; Qb_h = Qb_g + ab
        # FFN + residual + LN (eq16)
        Qa_t = self.ln_fa(Qa_h + self.ffn_a(Qa_h))
        Qb_t = self.ln_fb(Qb_h + self.ffn_b(Qb_h))
        y_pair = self.edge_cls(E).squeeze(-1)                       # [B,na,nb] pairwise logit (eq17)
        return Qa_t, Qb_t, E, y_pair


class TGR(nn.Module):
    """Triplet Graph Reasoning(§4.5): it/iv edge を relation node 化 → ontology グラフ → 三重項合成。"""

    def __init__(self, de=128, dh=128, heads=4):
        super().__init__()
        it_flat, iv_flat = _load_maps()
        self.register_buffer("it_flat", torch.tensor(it_flat))     # [85]
        self.register_buffer("iv_flat", torch.tensor(iv_flat))     # [85]
        self.psi_it = nn.Linear(de, dh); self.psi_iv = nn.Linear(de, dh)  # eq22
        # relation graph: it-node <-> iv-node (valid triplet で接続). dense GATv2 over 2部(it 180, iv 156)
        self.gat = BipartiteGATv2(dh, 1, dh, heads)                # edge attr は接続マスクのみ(de=1)
        self.ln = nn.LayerNorm(dh)
        # 三重項合成 z=[h_it; h_iv; h_it⊙h_iv; h_it-h_iv] -> logit (eq25,26)
        self.cls = nn.Sequential(nn.Linear(4 * dh, dh), nn.GELU(), nn.Linear(dh, 1))
        # ontology adjacency [180,156]: A[it,iv]=1 iff ∃triplet with that it&iv
        adj = torch.zeros(NIT, NIV)
        adj[it_flat, iv_flat] = 1.0
        self.register_buffer("adj", adj)

    def forward(self, E_it, E_iv):
        B = E_it.shape[0]
        Hit = self.psi_it(E_it.reshape(B, NIT, -1))                # [B,180,dh]
        Hiv = self.psi_iv(E_iv.reshape(B, NIV, -1))                # [B,156,dh]
        # relation graph message passing（adj を edge attr/mask に）
        edge = self.adj[None, :, :, None].expand(B, NIT, NIV, 1)   # [B,180,156,1]
        it2, iv2 = self.gat(Hit, Hiv, edge)
        # mask: 非接続ノードは message 0 になるよう softmax 前にマスクしたいが、
        # ここでは adj を edge attr として渡し LN 残差で安定化（valid のみ有効化）
        Hit = self.ln(Hit + it2 * (self.adj.sum(1, keepdim=True) > 0).float()[None])
        Hiv = self.ln(Hiv + iv2 * (self.adj.sum(0, keepdim=True).t() > 0).float()[None])
        # 三重項ごとに it/iv relation node を retrieve → 合成 (eq25)
        h_it = Hit[:, self.it_flat]                                # [B,85,dh]
        h_iv = Hiv[:, self.iv_flat]                                # [B,85,dh]
        z = torch.cat([h_it, h_iv, h_it * h_iv, (h_it - h_iv).abs()], dim=-1)  # [B,85,4dh]
        return self.cls(z).squeeze(-1)                             # [B,85] ivt logit


class SpiritFull(nn.Module):
    """TUF + PIC(I-T,I-V) + TGR。入力は凍結/学習 backbone の feats [B,T,N,d_bb]（heads を差し替え可）。"""

    def __init__(self, d_bb=1024, d=128, ds=256, de=128, do=256, dh=128, nhead=4, dropout=0.05, T=8, N=64,
                 mask_prior=False, frozen_feat=False):
        super().__init__()
        self.d, self.T, self.N = d, T, N
        self.mask_prior = mask_prior; self.frozen_feat = frozen_feat
        if mask_prior:  # #2: 器具マスクで keyframe 空間トークンを pool → 器具局在token を memory に追加
            self.inst_id_emb = nn.Embedding(NI, d)
        if frozen_feat:  # #3: 凍結 target(convnext)特徴 + presence を追加token に
            self.tgt_proj = nn.Linear(1024, d); self.pres_proj = nn.Linear(NI, d)
        self.vis_proj = nn.Linear(d_bb, d)
        self.register_buffer("spe", sinusoidal_pe(N, d), persistent=False)
        self.register_buffer("tpe", sinusoidal_pe(T, d), persistent=False)
        self.tuf_i = TUFBranch(NI, d, nhead, dropout=dropout)
        self.tuf_v = TUFBranch(NV, d, nhead, dropout=dropout)
        self.tuf_t = TUFBranch(NT, d, nhead, dropout=dropout)
        self.head_i = nn.Linear(d, 1); self.head_v = nn.Linear(d, 1); self.head_t = nn.Linear(d, 1)
        self.scene = nn.Sequential(nn.Linear(d, ds), nn.GELU())
        self.pic_it = PICBranch(NI, NT, d, ds, de, do, nhead, dropout)
        self.pic_iv = PICBranch(NI, NV, d, ds, de, do, nhead, dropout)
        self.tgr = TGR(de, dh, nhead)

    def forward(self, feats, imask=None, tfeat=None, pres=None):
        B, T, N, _ = feats.shape
        Z = self.vis_proj(feats) + self.spe.view(1, 1, N, self.d) + self.tpe.view(1, T, 1, self.d)
        M = Z.reshape(B, T * N, self.d)
        extra = []
        if self.mask_prior and imask is not None:  # #2: mask-guided 器具局在トークン
            g = int(round(N ** 0.5)); key = Z[:, -1]                          # keyframe tokens [B,N,d]
            im = torch.nn.functional.interpolate(imask, size=(g, g), mode="area").reshape(B, NI, N)
            imn = im / (im.sum(-1, keepdim=True) + 1e-4)
            extra.append(torch.einsum("bkn,bnd->bkd", imn, key) + self.inst_id_emb.weight[None])  # [B,12,d]
        if self.frozen_feat and tfeat is not None:  # #3: 凍結 target/presence トークン
            extra.append((self.tgt_proj(tfeat) + (self.pres_proj(pres) if pres is not None else 0)).unsqueeze(1))
        if extra:
            M = torch.cat([M] + extra, dim=1)
        G = Z.mean(dim=2); s = self.scene(M.mean(dim=1))
        Qi, Qv, Qt = self.tuf_i(M, G), self.tuf_v(M, G), self.tuf_t(M, G)
        oi = self.head_i(Qi).squeeze(-1); ov = self.head_v(Qv).squeeze(-1); ot = self.head_t(Qt).squeeze(-1)
        # PIC branches
        _, _, E_it, y_it = self.pic_it(Qi, Qt, s)                  # [B,ni,nt]
        _, _, E_iv, y_iv = self.pic_iv(Qi, Qv, s)                  # [B,ni,nv]
        # TGR -> ivt
        oivt = self.tgr(E_it, E_iv)                                # [B,85]
        return {"ivt": oivt, "i": oi, "v": ov, "t": ot,
                "it": y_it.reshape(B, NIT), "iv": y_iv.reshape(B, NIV)}


if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, N = 2, 8, 64
    m = SpiritFull(T=T, N=N).eval()
    with torch.no_grad():
        o = m(torch.randn(B, T, N, 1024))
    for k in ("ivt", "i", "v", "t", "it", "iv"):
        print(f"{k}: {tuple(o[k].shape)} nan={torch.isnan(o[k]).any().item()}")
    assert o["ivt"].shape == (B, 85) and o["it"].shape == (B, 180) and o["iv"].shape == (B, 156)
    print(f"params: {sum(p.numel() for p in m.parameters())/1e6:.2f}M")
    print("SPIRIT-FULL SMOKE OK")
