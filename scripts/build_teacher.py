"""Build the out-of-fold ensemble teacher used to distil the submitted students.

For each fold g we ensemble the three base models that held fold g out
(dense SPIRIT, dense Swin, keyframe SPIRIT), then concatenate the folds. Every frame is
therefore labelled by models that never trained on it, which is what makes the resulting
soft labels safe to distil from while still measuring CV honestly.

Fixed weights (1.0 / 0.4 / 0.7) — this ensemble scores 0.492 videowise ivt mAP.

Expected inputs in --oof_dir (one file per fold, from `--dump_oof ... --dump_split val`):
    spirit_dense_f{g}.npz   swin_dense_f{g}.npz   kfspirit_f{g}.npz
each holding `scores` [N,85], `vids` [N], `fids` [N].
"""
import argparse
import numpy as np


def load(path):
    d = np.load(path, allow_pickle=True)
    keys = [(str(v), int(f)) for v, f in zip(d["vids"], d["fids"])]
    return {k: i for i, k in enumerate(keys)}, d["scores"].astype("float32")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--w_dspirit", type=float, default=1.0)
    ap.add_argument("--w_dswin", type=float, default=0.4)
    ap.add_argument("--w_kfspirit", type=float, default=0.7)
    a = ap.parse_args()

    wsum = a.w_dspirit + a.w_dswin + a.w_kfspirit
    vids, fids, scores = [], [], []
    for g in range(a.folds):
        i_sp, s_sp = load(f"{a.oof_dir}/spirit_dense_f{g}.npz")
        i_sw, s_sw = load(f"{a.oof_dir}/swin_dense_f{g}.npz")
        i_kf, s_kf = load(f"{a.oof_dir}/kfspirit_f{g}.npz")
        common = sorted(set(i_sp) & set(i_sw) & set(i_kf))
        take = lambda idx, sc: sc[[idx[k] for k in common]]
        s = (a.w_dspirit * take(i_sp, s_sp)
             + a.w_dswin * take(i_sw, s_sw)
             + a.w_kfspirit * take(i_kf, s_kf)) / wsum
        scores.append(s)
        vids += [k[0] for k in common]
        fids += [k[1] for k in common]
        print(f"  fold{g}: {len(common)} frames")

    sc = np.concatenate(scores, 0).astype("float32")
    np.savez(a.out, scores=sc, vids=np.array(vids), fids=np.array(fids, dtype=np.int64))
    print(f"wrote {a.out}  {sc.shape}")


if __name__ == "__main__":
    main()
