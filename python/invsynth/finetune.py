"""
Phase 3 / Stage 2 - high-precision finetuner (NSGA-II).

  Stage-1 guess ──┬──> duration matcher (fits attack / sustain / gate / release
                  │     to the target's RMS envelope and file length)
                  v
     population = {guess, matched guess, gaussian neighbours}
                  │
     ┌─────────── generation loop ──────────────────────────────────────┐
     │ SBX crossover + polynomial mutation (continuous genes)           │
     │ uniform crossover + resampling (categorical genes)               │
     │ render every child through the SAME synth engine                 │
     │ objectives: f1 = multi-resolution STFT loss (timbre)             │
     │             f2 = RMS-envelope loss in dB (dynamics over time)    │
     │ non-dominated sort + crowding distance -> next population        │
     │ comparison window grows: selected 5 s -> 2x -> whole file        │
     └──────────── stop when scalar loss < tol or no progress ──────────┘

Two objectives instead of one weighted sum keeps "right timbre, wrong
envelope" and "right envelope, wrong timbre" solutions alive, which are often
one crossover away from the correct patch.
"""
from dataclasses import dataclass, field
import math
import os
from multiprocessing import Pool

import numpy as np

from . import engine
from . import features as F
from .params import CAT_CLASSES, CAT_IDX, CONT_IDX, IDX, N_PARAMS

FFT_SIZES = (512, 1024, 2048)
EPS = 1e-7


# ------------------------------------------------------------------ losses
def _mag(x, n_fft):
    hop = n_fft // 4
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    n_frames = 1 + (len(x) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    win = np.hanning(n_fft + 1)[:-1].astype(np.float32)
    return np.abs(np.fft.rfft(x[idx] * win, axis=1)).astype(np.float32)


def rms_env_db(x, hop=F.HOP):
    n = max(1, len(x) // hop)
    fr = x[:n * hop].reshape(n, hop)
    db = 10 * np.log10(np.mean(fr * fr, axis=1) + 1e-12)
    return np.maximum(db - db.max(), -60.0)


class Target:
    """Pre-computed target representations for one comparison region."""

    def __init__(self, x):
        self.x = F.peak_normalize_f(x)
        self.n = len(x)
        self.mags = {n: _mag(self.x, n) for n in FFT_SIZES}
        self.env = rms_env_db(self.x)

    def objectives(self, y):
        f1 = 0.0
        for n in FFT_SIZES:
            T, R = self.mags[n], _mag(y, n)
            sc = np.linalg.norm(T - R) / (np.linalg.norm(T) + EPS)
            lm = np.mean(np.abs(np.log(T + EPS) - np.log(R + EPS)))
            f1 += sc + 0.1 * lm
        f1 /= len(FFT_SIZES)
        f2 = float(np.mean(np.abs(self.env - rms_env_db(y)[:len(self.env)]))) / 60.0
        return np.array([f1, f2])


def scalar(obj):
    return float(obj[0] + 0.5 * obj[1])


# ------------------------------------------------------ duration matching
def _time_to_norm(t):
    return float(np.clip(math.log(max(t, 1e-3) / 0.001) / math.log(10000.0), 0, 1))


def match_duration(p, x_note, sr=F.SR):
    """
    Fit amp attack / sustain / gate / release so the rendered note has the
    same length and loudness contour as the target note (x_note starts at
    note-on and runs to the end of the region that will be compared).
    """
    p = p.copy()
    env = rms_env_db(x_note)
    fps = sr / F.HOP
    D = len(x_note) / sr
    tp = int(np.argmax(env >= -1.0))
    p[IDX["aenv_attack"]] = _time_to_norm(tp / fps)
    loud = np.nonzero(env > -12.0)[0]
    tg = int(loud[-1]) if len(loud) else tp
    audible = np.nonzero(env > -59.0)[0]
    te = int(audible[-1]) if len(audible) else len(env) - 1
    if te >= len(env) - 2 and tg >= len(env) - 3:
        p[IDX["gate"]] = 1.0                          # still held at end of file
    else:
        p[IDX["gate"]] = float(np.clip((tg / fps / D - 0.05) / 0.95, 0, 1))
        seg = env[tg:te + 1]
        if len(seg) >= 3:
            slope = np.polyfit(np.arange(len(seg)) / fps, seg, 1)[0]   # dB/s
            if slope < -1e-3:
                tau = 8.686 / -slope
                p[IDX["aenv_release"]] = _time_to_norm(5.0 * tau)
    if tg > tp + int(0.2 * fps):
        p[IDX["aenv_sustain"]] = float(np.clip(10 ** (np.median(env[tp:tg + 1]) / 20), 0, 1))
    return p


# ------------------------------------------------------ parallel evaluation
_G = {}


def _init(target_x, n_full_note, sr):
    _G["t"] = Target(target_x)
    _G["n_full"] = n_full_note
    _G["sr"] = sr


def _gate(p, n_full):
    return int(n_full * (0.05 + 0.95 * float(p[IDX["gate"]])))


def _eval(p):
    t = _G["t"]
    y = engine.render_gated(p, t.n, _gate(p, _G["n_full"]), _G["sr"])
    if not np.all(np.isfinite(y)) or np.max(np.abs(y)) < 1e-6:
        return np.array([10.0, 10.0])
    return t.objectives(y)


# ------------------------------------------------------------ NSGA-II
def nondominated_sort(obj):
    n = len(obj)
    S = [[] for _ in range(n)]
    cnt = np.zeros(n, dtype=int)
    rank = np.zeros(n, dtype=int)
    fronts = [[]]
    for i in range(n):
        for j in range(n):
            if np.all(obj[i] <= obj[j]) and np.any(obj[i] < obj[j]):
                S[i].append(j)
            elif np.all(obj[j] <= obj[i]) and np.any(obj[j] < obj[i]):
                cnt[i] += 1
        if cnt[i] == 0:
            fronts[0].append(i)
    k = 0
    while fronts[k]:
        nxt = []
        for i in fronts[k]:
            for j in S[i]:
                cnt[j] -= 1
                if cnt[j] == 0:
                    rank[j] = k + 1
                    nxt.append(j)
        k += 1
        fronts.append(nxt)
    return fronts[:-1], rank


def crowding(obj, front):
    d = np.zeros(len(front))
    if len(front) <= 2:
        return d + np.inf
    f = obj[front]
    for m in range(f.shape[1]):
        o = np.argsort(f[:, m])
        d[o[0]] = d[o[-1]] = np.inf
        span = f[o[-1], m] - f[o[0], m] + 1e-12
        d[o[1:-1]] += (f[o[2:], m] - f[o[:-2], m]) / span
    return d


def _sbx(a, b, rng, eta=15.0):
    u = rng.random(len(a))
    beta = np.where(u <= 0.5, (2 * u) ** (1 / (eta + 1)), (1 / (2 * (1 - u))) ** (1 / (eta + 1)))
    c1 = 0.5 * ((1 + beta) * a + (1 - beta) * b)
    c2 = 0.5 * ((1 - beta) * a + (1 + beta) * b)
    return np.clip(c1, 0, 1), np.clip(c2, 0, 1)


def _poly_mut(x, rng, pm, eta=20.0):
    x = x.copy()
    for i in np.nonzero(rng.random(len(x)) < pm)[0]:
        u = rng.random()
        d = (2 * u) ** (1 / (eta + 1)) - 1 if u < 0.5 else 1 - (2 * (1 - u)) ** (1 / (eta + 1))
        x[i] = np.clip(x[i] + d, 0, 1)
    return x


CONT = np.array(CONT_IDX)


def make_children(pop, rank, crowd, rng, cat_mut=0.05):
    n = len(pop)
    kids = []

    def tour():
        i, j = rng.integers(n, size=2)
        if rank[i] != rank[j]:
            return pop[i] if rank[i] < rank[j] else pop[j]
        return pop[i] if crowd[i] > crowd[j] else pop[j]

    while len(kids) < n:
        a, b = tour(), tour()
        c1, c2 = a.copy(), b.copy()
        if rng.random() < 0.9:
            c1[CONT], c2[CONT] = _sbx(a[CONT], b[CONT], rng)
        for idx, k in zip(CAT_IDX, CAT_CLASSES):
            if rng.random() < 0.5:
                c1[idx], c2[idx] = c2[idx], c1[idx]
            for c in (c1, c2):
                if rng.random() < cat_mut:
                    c[idx] = (rng.integers(k) + 0.5) / k
        for c in (c1, c2):
            c[CONT] = _poly_mut(c[CONT], rng, 2.0 / len(CONT))
            kids.append(c)
    return np.array(kids[:n])


@dataclass
class FinetuneConfig:
    pop: int = 48
    generations: int = 60
    tol: float = 0.02               # stop when scalar loss below this
    patience: int = 15
    window_mode: str = "progressive"   # "selected" | "full" | "progressive"
    max_compare_sec: float = 30.0   # hard cap on compared region (keeps cost bounded)
    workers: int = max(1, (os.cpu_count() or 2) - 1)
    seed: int = 0


@dataclass
class FinetuneResult:
    params: np.ndarray
    loss: float
    history: list = field(default_factory=list)
    pareto: np.ndarray = None


def _stage_ends(w: F.Window, cfg: FinetuneConfig, sr):
    """Comparison region end (samples from note-on) for each third of the run."""
    note_len = w.total - w.start
    cap = int(cfg.max_compare_sec * sr)
    sel = min(w.length, note_len)
    full = min(note_len, cap)
    if cfg.window_mode == "selected":
        return [sel] * 3
    if cfg.window_mode == "full":
        return [full] * 3
    return [sel, min(max(2 * sel, sel), full), full]


def finetune(x: np.ndarray, p0: np.ndarray, w: F.Window, cfg=FinetuneConfig(),
             sr=F.SR, progress=None) -> FinetuneResult:
    """
    x: full target (peak-normalised, SR). p0: stage-1 prediction. w: the
    analysis window from features.select_window. The note is assumed to start
    at w.start (the snapped onset).
    """
    rng = np.random.default_rng(cfg.seed)
    note = x[w.start:]
    n_full = len(note)
    ends = _stage_ends(w, cfg, sr)

    matched = match_duration(p0, note[:min(n_full, int(cfg.max_compare_sec * sr))], sr)
    pop = [p0.copy(), matched]
    while len(pop) < cfg.pop:
        base = pop[rng.integers(2)].copy()
        base[CONT] = np.clip(base[CONT] + rng.normal(0, 0.05, len(CONT)), 0, 1)
        for idx, k in zip(CAT_IDX, CAT_CLASSES):
            if rng.random() < 0.1:
                base[idx] = (rng.integers(k) + 0.5) / k
        pop.append(base)
    pop = np.array(pop)

    best_p, best_l, stall, hist = pop[0], float("inf"), 0, []
    gens_per_stage = max(1, cfg.generations // 3)
    obj = None
    stage = -1
    pool = None
    try:
        for g in range(cfg.generations):
            s = min(g // gens_per_stage, 2)
            if s != stage:                         # comparison window grows
                stage = s
                if pool:
                    pool.close(); pool.join()
                seg = note[:ends[s]]
                pool = Pool(cfg.workers, initializer=_init, initargs=(seg, n_full, sr))
                if s > 0:                          # re-fit release to the longer region
                    pop[-1] = match_duration(best_p, seg, sr)
                obj = np.array(pool.map(_eval, list(pop)))
                best_l = float("inf")
            fronts, rank = nondominated_sort(obj)
            crowd = np.zeros(len(pop))
            for fr in fronts:
                crowd[fr] = crowding(obj, fr)
            kids = make_children(pop, rank, crowd, rng)
            kobj = np.array(pool.map(_eval, list(kids)))
            allp, allo = np.vstack([pop, kids]), np.vstack([obj, kobj])
            fronts, _ = nondominated_sort(allo)
            sel = []
            for fr in fronts:
                if len(sel) + len(fr) <= cfg.pop:
                    sel += fr
                else:
                    cd = crowding(allo, fr)
                    sel += [fr[i] for i in np.argsort(-cd)[:cfg.pop - len(sel)]]
                    break
            pop, obj = allp[sel], allo[sel]
            sc = np.array([scalar(o) for o in obj])
            i = int(np.argmin(sc))
            if sc[i] < best_l - 1e-4:
                best_l, best_p, stall = float(sc[i]), pop[i].copy(), 0
            else:
                stall += 1
            hist.append(dict(gen=g, stage=s, best=best_l, window_sec=ends[s] / sr))
            if progress:
                progress(g, cfg.generations, best_l)
            if s == 2 and (best_l < cfg.tol or stall >= cfg.patience):
                break
    finally:
        if pool:
            pool.close(); pool.join()
    front0 = pop[nondominated_sort(obj)[0][0]]
    return FinetuneResult(best_p, best_l, hist, front0)
