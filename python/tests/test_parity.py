#!/usr/bin/env python3
"""
Train/deploy parity: the network is trained on the Python engine but the
plugin plays and optimises with the C++ engine. Run after any DSP change:

    python tests/test_parity.py path/to/invsynth_cli
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from invsynth import engine, features as F, params as P  # noqa: E402

CLI = sys.argv[1] if len(sys.argv) > 1 else "invsynth_cli"


def run(*args):
    return subprocess.run([CLI, *map(str, args)], check=True, capture_output=True, text=True).stdout


def test_engine(n_patches=40, seconds=2.0):
    rng = np.random.default_rng(1)
    worst = 0.0
    with tempfile.TemporaryDirectory() as d:
        for k in range(n_patches):
            p = P.random_params(rng)
            n = int(seconds * F.SR)
            g = engine.gate_samples(p, n)
            ref = engine.render_gated(p, n, g, F.SR, normalize=False)
            np.savetxt(os.path.join(d, "p.txt"), p[None], fmt="%.9g")
            run("render", os.path.join(d, "p.txt"), n, g, os.path.join(d, "o.f32"))
            out = np.fromfile(os.path.join(d, "o.f32"), dtype=np.float32)
            err = np.max(np.abs(out - ref)) / (np.max(np.abs(ref)) + 1e-9)
            worst = max(worst, err)
            assert err < 1e-3, f"patch {k}: relative error {err:.2e}"
    print(f"engine parity OK ({n_patches} patches, worst rel. err {worst:.1e})")


def test_features():
    rng = np.random.default_rng(2)
    x = F.peak_normalize_f(engine.render(P.random_params(rng), 3.0))
    with tempfile.TemporaryDirectory() as d:
        x.astype(np.float32).tofile(os.path.join(d, "x.f32"))
        run("mel", os.path.join(d, "x.f32"), os.path.join(d, "m.f32"))
        out = np.fromfile(os.path.join(d, "m.f32"), dtype=np.float32).reshape(F.N_MELS, -1)
    ref = F.log_mel(x)
    err = np.max(np.abs(out - ref))
    assert out.shape == ref.shape and err < 2e-3, (out.shape, ref.shape, err)
    print(f"mel parity OK (max abs err {err:.1e})")


def test_window():
    rng = np.random.default_rng(3)
    # 12 s file: quiet pad then a dense patch starting at ~6 s
    quiet = 0.01 * engine.render(P.random_params(rng), 6.0)
    pl = P.random_params(rng)
    pl[P.IDX["gate"]] = 1.0
    pl[P.IDX["aenv_sustain"]] = 0.9
    loud = engine.render(pl, 6.0)
    x = F.peak_normalize_f(np.concatenate([quiet, loud]))
    with tempfile.TemporaryDirectory() as d:
        x.tofile(os.path.join(d, "x.f32"))
        s, l, t = map(int, run("window", os.path.join(d, "x.f32")).split())
    w = F.select_window(x)
    assert (s, l, t) == (w.start, w.length, w.total), ((s, l, t), w)
    assert abs(s / F.SR - 6.0) < 1.1, s / F.SR
    print(f"window parity OK (start {s / F.SR:.2f}s)")


if __name__ == "__main__":
    test_engine()
    test_features()
    test_window()
