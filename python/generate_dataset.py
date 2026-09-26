#!/usr/bin/env python3
"""
Phase 2 - variable-length dataset generation.

  random params ─> render 1..10 s note ─> (augment) ─> window selector
        │                                                  │
        └──────────────── Y (41 floats) ──────┐            v
                                              ├──> shard: mel crop (128 x T, fp16)
                                ctx (3) ──────┘            + offsets, ctx, params

Each shard is a folder of .npy files that are opened with mmap at training
time, so a dataset of millions of variable-length examples never has to fit
in RAM and needs no padding on disk.

Usage:
    python generate_dataset.py --out data/train --n 200000 --workers 8
    python generate_dataset.py --out data/val   --n 5000   --seed 999
"""
import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np
from scipy.signal import lfilter

from invsynth import engine, features as F, params as P


def augment(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Light domain randomisation so the model tolerates real recordings:
    background noise, spectral tilt (mic/room colouration) and gain.
    Labels are untouched because none of these are synth parameters.
    """
    if rng.random() < 0.5:
        x = x + rng.normal(0, 10 ** (rng.uniform(-70, -35) / 20), x.shape).astype(np.float32)
    if rng.random() < 0.5:
        a = rng.uniform(0.05, 0.6)                      # one-pole tilt filter
        lp = lfilter([a], [1.0, a - 1.0], x).astype(np.float32)
        mix = rng.uniform(-0.5, 0.5)
        x = x + mix * (lp - x) if mix > 0 else x - mix * (x - lp)
    return F.peak_normalize_f(x.astype(np.float32))


def make_example(rng: np.random.Generator):
    for _ in range(20):                                  # retry silent patches
        y = P.random_params(rng)
        dur = float(np.exp(rng.uniform(np.log(1.0), np.log(10.0))))
        x = engine.render(y, dur, F.SR)
        if np.max(np.abs(x)) < 1e-4 or not np.all(np.isfinite(x)):
            continue
        if rng.random() < 0.5:
            x = augment(x, rng)
        mel, ctx, w = F.prepare_input(x)
        if mel.max() < -0.6:                             # window is ~silent
            continue
        return mel.astype(np.float16), ctx, y, dur
    return None


def build_shard(args):
    out_dir, shard_id, n, seed = args
    rng = np.random.default_rng(seed)
    mels, ctxs, ys, durs = [], [], [], []
    while len(ys) < n:
        ex = make_example(rng)
        if ex is None:
            continue
        m, c, y, d = ex
        mels.append(m); ctxs.append(c); ys.append(y); durs.append(d)
    lengths = np.array([m.shape[1] for m in mels], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    d = os.path.join(out_dir, f"shard_{shard_id:05d}")
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, "mels.npy"), np.concatenate(mels, axis=1))   # (128, sumT)
    np.save(os.path.join(d, "offsets.npy"), offsets)
    np.save(os.path.join(d, "ctx.npy"), np.stack(ctxs))
    np.save(os.path.join(d, "params.npy"), np.stack(ys).astype(np.float32))
    np.save(os.path.join(d, "durations.npy"), np.array(durs, dtype=np.float32))
    return shard_id, len(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--shard-size", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    n_shards = (a.n + a.shard_size - 1) // a.shard_size
    jobs = [(a.out, i, min(a.shard_size, a.n - i * a.shard_size), a.seed * 100_003 + i)
            for i in range(n_shards)]
    engine.render(P.DEFAULTS, 0.1)                       # trigger JIT compile once
    t0 = time.time()
    with Pool(a.workers) as pool:
        for sid, cnt in pool.imap_unordered(build_shard, jobs):
            print(f"shard {sid:05d}: {cnt} examples  ({time.time() - t0:.0f}s)")
    meta = dict(n=a.n, n_params=P.N_PARAMS, sr=F.SR, n_fft=F.N_FFT, hop=F.HOP,
                n_mels=F.N_MELS, window_sec=F.WINDOW_SEC,
                param_names=[p.name for p in P.PARAMS])
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
