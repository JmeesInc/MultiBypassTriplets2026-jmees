"""Logit-ensemble evaluation of N trained fusion checkpoints on the val fold.

Each (config, checkpoint) is run with its OWN clip sampling (stride2 vs logsym); sigmoid scores
are cached to results/{run}/fold0/val_scores.npy (compute once, reuse). The val item order is
identical across configs (same folds_csv/val_fold/drop_empty), so cached score arrays align and
any subset can be averaged instantly.

Usage:
  # compute (and cache) + evaluate a specific ensemble:
  GPU=1 python src/ensemble_eval.py --runs verb_swin3db_fusion_v5_preB verb_swin3db_fusion_v6_logsym_oldpre
  # sweep all pairs/triples over the cached members (no GPU needed once cached):
  python src/ensemble_eval.py --runs A B C D --sweep
"""
import argparse
import itertools
import os

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

import train_verb_fusion as F

HERE = os.path.dirname(__file__)


def rundir(run):
    return os.path.join(HERE, "..", "results", run, "fold0")


@torch.no_grad()
def compute_scores(cfg, ckpt, device):
    v2f = F.load_folds(cfg["data"]["folds_csv"])
    va = F.build_index(cfg, v2f, True)
    vl = DataLoader(F.ClipDS(va, cfg, False), batch_size=cfg["train"]["bs"], shuffle=False,
                    num_workers=cfg["train"]["workers"], pin_memory=True)
    mcfg = cfg.get("model", {})
    model = F.SwinFusion(use_target=mcfg.get("use_target", True),
                         use_instrument=mcfg.get("use_instrument", True),
                         fusion=mcfg.get("fusion", "concat")).to(device).eval()
    model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    S = []
    for b in vl:
        with torch.autocast("cuda", dtype=torch.float16):
            oivt, *_ = model(b["clip"].to(device), b["tfeat"].to(device), b["tlogit"].to(device),
                             b["pres"].to(device), b["imask"].to(device))
        S.append(torch.sigmoid(oivt).float().cpu().numpy())
    return np.concatenate(S), va


def get_scores(run, ckpt_name, device, force=False):
    """Load cached val_scores.npy or compute+cache it. Returns (P, va)."""
    npy = os.path.join(rundir(run), "val_scores.npy")
    cfg = yaml.safe_load(open(os.path.join(rundir(run), "config.yaml")))
    v2f = F.load_folds(cfg["data"]["folds_csv"])
    va = F.build_index(cfg, v2f, True)
    if os.path.exists(npy) and not force:
        P = np.load(npy)
        if P.shape[0] == len(va):
            return P, va
    P, va = compute_scores(cfg, os.path.join(rundir(run), ckpt_name), device)
    np.save(npy, P)
    return P, va


def report(P, va, tag):
    Yivt = np.stack([it[3] for it in va]); vids = [it[0] for it in va]
    pi, pv, pt = F.ivt_to_comp(P)
    Yi = np.stack([it[4] for it in va]); Yv = np.stack([it[5] for it in va]); Yt = np.stack([it[6] for it in va])
    m = {"ivt": F.videowise_map(P, Yivt, vids, F.NIVT), "i": F.videowise_map(pi, Yi, vids, F.NI),
         "v": F.videowise_map(pv, Yv, vids, F.NV), "t": F.videowise_map(pt, Yt, vids, F.NT)}
    print(f"{tag:52s} ivt={m['ivt']:.4f} i={m['i']:.4f} v={m['v']:.4f} t={m['t']:.4f}", flush=True)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="result dir names under results/")
    ap.add_argument("--ckpt", default="best_model.pth")
    ap.add_argument("--force", action="store_true", help="recompute scores even if cached")
    ap.add_argument("--sweep", action="store_true", help="evaluate all subsets (size>=2) of --runs")
    args = ap.parse_args()
    device = "cuda"
    Ps, ref_va = {}, None
    for r in args.runs:
        P, va = get_scores(r, args.ckpt, device, args.force)
        if ref_va is None:
            ref_va = va
        else:
            assert [it[:2] for it in va] == [it[:2] for it in ref_va], f"val order mismatch: {r}"
        report(P, va, f"[single] {r}")
        Ps[r] = P
    if args.sweep:
        print("--- subset sweep (by ivt) ---", flush=True)
        results = []
        for k in range(2, len(args.runs) + 1):
            for combo in itertools.combinations(args.runs, k):
                ens = np.mean([Ps[r] for r in combo], axis=0)
                m = report(ens, ref_va, f"[ens {k}] " + "+".join(c.replace('verb_swin3db_fusion_', '') for c in combo))
                results.append((m["ivt"], combo))
        best = max(results)
        print(f"\nBEST: ivt={best[0]:.4f} <- {best[1]}", flush=True)
    else:
        ens = np.mean([Ps[r] for r in args.runs], axis=0)
        report(ens, ref_va, f"[ENSEMBLE mean of {len(args.runs)}]")


if __name__ == "__main__":
    main()
