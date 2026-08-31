"""4-fold CV of the 2-student ensemble (videowise ivt mAP).

Videowise = average precision is computed per video over the classes present in that
video, then averaged across videos — matching the challenge metric. Reproduces 0.5181
for the submitted students at --w_swin 0.3.
"""
import argparse
from collections import defaultdict

import numpy as np
from sklearn.metrics import average_precision_score

NIVT = 85


def load(path):
    d = np.load(path, allow_pickle=True)
    idx = {(str(v), int(f)): i for i, (v, f) in enumerate(zip(d["vids"], d["fids"]))}
    return idx, d["scores"].astype("float32"), d["labels"].astype("float32")


def videowise_map(scores, labels, video_of):
    by_video = defaultdict(list)
    for i, v in enumerate(video_of):
        by_video[v].append(i)
    per_video = []
    for _, rows in by_video.items():
        aps = [average_precision_score(labels[rows][:, c], scores[rows][:, c])
               for c in range(NIVT) if labels[rows][:, c].sum() > 0]
        per_video.append(np.mean(aps))
    return float(np.mean(per_video))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof_dir", required=True)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--w_swin", type=float, default=0.3)
    a = ap.parse_args()

    sp_all, sw_all, ens_all = [], [], []
    for f in range(a.folds):
        i_sp, s_sp, lab = load(f"{a.oof_dir}/spirit_ens_oof_f{f}.npz")
        i_sw, s_sw, _ = load(f"{a.oof_dir}/swin_ens_oof_f{f}.npz")
        common = sorted(set(i_sp) & set(i_sw))
        SP = s_sp[[i_sp[k] for k in common]]
        SW = s_sw[[i_sw[k] for k in common]]
        L = lab[[i_sp[k] for k in common]]
        V = [k[0] for k in common]
        s_only = videowise_map(SP, L, V)
        w_only = videowise_map(SW, L, V)
        ens = videowise_map((1 - a.w_swin) * SP + a.w_swin * SW, L, V)
        sp_all.append(s_only); sw_all.append(w_only); ens_all.append(ens)
        print(f"  f{f}: SPIRIT={s_only:.4f}  Swin={w_only:.4f}  ensemble={ens:.4f}")

    print(f"\n  SPIRIT student CV : {np.mean(sp_all):.4f}")
    print(f"  Swin student CV   : {np.mean(sw_all):.4f}")
    print(f"  ENSEMBLE CV       : {np.mean(ens_all):.4f}  (w_swin={a.w_swin})")


if __name__ == "__main__":
    main()
