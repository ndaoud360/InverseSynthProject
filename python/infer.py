#!/usr/bin/env python3
"""
End-to-end offline inverse synthesis from the command line.

    python infer.py sample.wav --model inverse_synth.onnx --out match
    python infer.py long_pad.wav --model inverse_synth.onnx --window-mode full

Writes match.json (normalised + named parameters) and match.wav (rendered).
The generated JSON can be loaded by the plugin ("Load preset").
"""
import argparse
import json
import time

import numpy as np
import onnxruntime as ort

from invsynth import engine, features as F, params as P
from invsynth.finetune import FinetuneConfig, finetune


def predict(sess, mel, ctx):
    return sess.run(None, {"mel": mel[None].astype(np.float32),
                           "ctx": ctx[None].astype(np.float32)})[0][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="match")
    ap.add_argument("--window-mode", default="progressive",
                    choices=["selected", "full", "progressive"])
    ap.add_argument("--generations", type=int, default=60)
    ap.add_argument("--pop", type=int, default=48)
    ap.add_argument("--no-finetune", action="store_true")
    a = ap.parse_args()

    x = F.peak_normalize_f(F.load_audio(a.audio))
    print(f"loaded {len(x) / F.SR:.2f}s")
    t0 = time.time()
    mel, ctx, w = F.prepare_input(x)
    print(f"window {w.start / F.SR:.2f}s + {w.length / F.SR:.2f}s  mel {mel.shape}")

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    sess = ort.InferenceSession(a.model, opts, providers=["CPUExecutionProvider"])
    p = predict(sess, mel, ctx)
    print(f"stage 1: {1000 * (time.time() - t0):.0f} ms")

    if not a.no_finetune:
        cfg = FinetuneConfig(pop=a.pop, generations=a.generations, window_mode=a.window_mode)
        res = finetune(x, p, w, cfg,
                       progress=lambda g, n, l: print(f"  gen {g + 1}/{n}  loss {l:.4f}", end="\r"))
        print(f"\nstage 2: loss {res.loss:.4f}  ({time.time() - t0:.1f}s total)")
        p = res.params

    note = len(x) - w.start
    y = engine.render_gated(p, note, int(note * (0.05 + 0.95 * p[P.IDX["gate"]])), F.SR)
    import soundfile as sf
    sf.write(a.out + ".wav", y, F.SR)
    with open(a.out + ".json", "w") as f:
        json.dump({"version": 1, "normalized": [float(v) for v in p],
                   "named": {q.name: float(v) for q, v in zip(P.PARAMS, p)}}, f, indent=2)
    print("wrote", a.out + ".json", a.out + ".wav")


if __name__ == "__main__":
    main()
