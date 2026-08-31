"""SPIRIT fine-tune 版: DINOv3-L backbone を統合（下位block凍結・上位blockのみ学習）+ SpiritStageA head。

論文 §5.3 "backbone frozen up to block 15" に準拠: blocks[0:freeze] 凍結, blocks[freeze:] 学習。
メモリ効率: 凍結blockは torch.no_grad() で活性化を保存しない → 上位blockのみ backprop。
入力は raw frames [B,T,3,224,224]（凍結キャッシュでなく生frame → RandAugment 可・backbone学習可）。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt

from model_spirit import SpiritStageA, NI, NV, NT, NIVT  # noqa: F401  (heads/dims 再利用)

POOL = 8


class DinoBackboneTrunk(nn.Module):
    """DINOv3 ViT を包み、下位block凍結・上位block学習で patch token(8x8 pool) を返す。"""

    def __init__(self, backbone, freeze_blocks=16, grad_ckpt=True):
        super().__init__()
        self.bb = backbone
        self.nblk = len(backbone.blocks)
        self.freeze = freeze_blocks
        self.grad_ckpt = grad_ckpt
        # 凍結設定: 全部凍結 → 上位blockとnormのみ解凍
        for p in self.bb.parameters():
            p.requires_grad_(False)
        for i in range(freeze_blocks, self.nblk):
            for p in self.bb.blocks[i].parameters():
                p.requires_grad_(True)
        for p in self.bb.norm.parameters():
            p.requires_grad_(True)

    def forward(self, imgs):
        """imgs [BT,3,224,224] -> patch tokens pooled [BT,64,D]."""
        bb = self.bb
        x, (H, W) = bb.prepare_tokens_with_masks(imgs)   # [BT, 1+nstore+Npatch, D]
        xl = [x]
        for i, blk in enumerate(bb.blocks):
            rope = [bb.rope_embed(H=H, W=W)] if bb.rope_embed is not None else [None]
            if i < self.freeze:
                with torch.no_grad():                    # 凍結: 活性化保存せず backprop 遮断
                    xl = blk(xl, rope)
                xl = [xl[0].detach()]
            elif self.grad_ckpt and self.training:
                xl = [ckpt.checkpoint(lambda t, r=rope, b=blk: b([t], r)[0], xl[0], use_reentrant=False)]
            else:
                xl = blk(xl, rope)
        x = bb.norm(xl[0])
        patch = x[:, bb.n_storage_tokens + 1:]           # [BT, Npatch, D]
        BT, Np, C = patch.shape
        g = int(round(Np ** 0.5))
        p = patch.transpose(1, 2).reshape(BT, C, g, g)
        p = F.adaptive_avg_pool2d(p, (POOL, POOL)).reshape(BT, C, POOL * POOL).transpose(1, 2)  # [BT,64,C]
        return p


class SpiritStageA_FT(nn.Module):
    """backbone trunk + SpiritStageA head。forward は raw frames [B,T,3,224,224]。"""

    def __init__(self, backbone, freeze_blocks=16, grad_ckpt=True,
                 d=128, ds=256, nhead=4, dropout=0.05, T=8, N=64):
        super().__init__()
        self.trunk = DinoBackboneTrunk(backbone, freeze_blocks, grad_ckpt)
        self.head = SpiritStageA(d_bb=1024, d=d, ds=ds, nhead=nhead, dropout=dropout, T=T, N=N)
        self.T = T

    def forward(self, frames):
        B, T = frames.shape[:2]
        feats = self.trunk(frames.reshape(B * T, *frames.shape[2:]))   # [B*T,64,1024]
        feats = feats.reshape(B, T, feats.shape[1], feats.shape[2])    # [B,T,64,1024]
        return self.head(feats)

    def param_groups(self, head_lr, backbone_lr):
        bb = [p for p in self.trunk.parameters() if p.requires_grad]
        hd = [p for p in self.head.parameters() if p.requires_grad]
        return [{"params": bb, "lr": backbone_lr}, {"params": hd, "lr": head_lr}]


class SpiritFull_FT(nn.Module):
    """フル SPIRIT(TUF+PIC+TGR) + fine-tune backbone。Stage B/C/D 用（ivt は TGR 出力, it/iv head 有）。"""

    def __init__(self, backbone, freeze_blocks=16, grad_ckpt=False,
                 d=128, ds=256, de=128, do=256, dh=128, nhead=4, dropout=0.05, T=8, N=64,
                 mask_prior=False, frozen_feat=False):
        super().__init__()
        from model_spirit_full import SpiritFull
        self.trunk = DinoBackboneTrunk(backbone, freeze_blocks, grad_ckpt)
        self.head = SpiritFull(d_bb=1024, d=d, ds=ds, de=de, do=do, dh=dh, nhead=nhead, dropout=dropout, T=T, N=N,
                               mask_prior=mask_prior, frozen_feat=frozen_feat)
        self.T = T

    def forward(self, frames, imask=None, tfeat=None, pres=None):
        B, T = frames.shape[:2]
        feats = self.trunk(frames.reshape(B * T, *frames.shape[2:]))
        feats = feats.reshape(B, T, feats.shape[1], feats.shape[2])
        return self.head(feats, imask=imask, tfeat=tfeat, pres=pres)

    def param_groups(self, head_lr, backbone_lr):
        bb = [p for p in self.trunk.parameters() if p.requires_grad]
        hd = [p for p in self.head.parameters() if p.requires_grad]
        return [{"params": bb, "lr": backbone_lr}, {"params": hd, "lr": head_lr}]
