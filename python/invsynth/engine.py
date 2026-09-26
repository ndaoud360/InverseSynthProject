"""
Reference synth engine (Python / Numba).

This is a line-by-line mirror of cpp/core/SynthVoice.cpp + Effects.cpp.
Training data is generated with THIS engine and the plugin plays with the C++
engine, so any divergence becomes a train/deploy mismatch. Keep them in sync.

Signal flow (mono):

  LFO ─┬─> pitch ─────────────┐
       ├─> wavetable pos      │
       ├─> cutoff             │
       └─> amp (tremolo)      │
                              v
  OSC2 (master) ──sync──> OSC1 (phase-modulated by OSC2 = FM)
     │                      │
     └──────> mixer <───────┘   (level mix  ⊕  ring mod)
                 │
          SVF LP/BP/HP  <── Filter ADSR (+/- 6 oct)
                 │
             Amp ADSR
                 │
      Distortion → Chorus → Reverb → out
"""
import math
import numpy as np

try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    HAVE_NUMBA = False
    import warnings
    warnings.warn("numba not installed: rendering will be ~100x slower")

    def njit(*a, **k):
        if a and callable(a[0]):
            return a[0]
        return lambda f: f

from .params import N_PARAMS

WT_FRAMES = 8
WT_SIZE = 2048
WT_HARMONICS = 64
CTRL_RATE = 16          # filter coefficients recomputed every 16 samples


def make_wavetables() -> np.ndarray:
    """8 procedurally-generated morphing frames (identical formula in C++)."""
    n = np.arange(WT_SIZE) / WT_SIZE
    wt = np.zeros((WT_FRAMES, WT_SIZE), dtype=np.float64)
    for k in range(WT_FRAMES):
        for h in range(1, WT_HARMONICS + 1):
            a = h ** (-(0.6 + 0.2 * k))
            if h % 2 == 0:
                a *= 1.0 - k / (WT_FRAMES - 1)
            a *= 1.0 + 0.5 * math.sin(h * k * 0.7)
            wt[k] += a * np.sin(2 * np.pi * h * n)
        wt[k] /= np.max(np.abs(wt[k]))
    return wt.astype(np.float32)


WAVETABLES = make_wavetables()


# --------------------------------------------------------------------------- #
# Kernels
# --------------------------------------------------------------------------- #
@njit(cache=True)
def _cat(v, n):
    k = int(v * n)
    return n - 1 if k > n - 1 else (0 if k < 0 else k)


@njit(cache=True)
def _env_time(v):
    return 0.001 * 10000.0 ** v          # 1 ms .. 10 s


@njit(cache=True)
def _rand(state):
    """LCG, returns (new_state, value in [-1,1)). Same constants in C++."""
    state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
    return state, state / 2147483648.0 - 1.0


@njit(cache=True)
def _blep(t, dt):
    if t < dt:
        t /= dt
        return t + t - t * t - 1.0
    elif t > 1.0 - dt:
        t = (t - 1.0) / dt
        return t * t + t + t + 1.0
    return 0.0


@njit(cache=True)
def _osc(wave, t, dt, wtpos, noise, wt):
    if wave == 0:
        return math.sin(2.0 * math.pi * t)
    elif wave == 1:
        return 2.0 * t - 1.0 - _blep(t, dt)
    elif wave == 2:
        t2 = t + 0.5
        t2 -= math.floor(t2)
        s = 1.0 if t < 0.5 else -1.0
        return s + _blep(t, dt) - _blep(t2, dt)
    elif wave == 3:
        return 1.0 - 4.0 * abs(t - 0.5)
    elif wave == 4:
        return noise
    else:
        x = wtpos * (WT_FRAMES - 1)
        f0 = int(x)
        if f0 > WT_FRAMES - 2:
            f0 = WT_FRAMES - 2
        fr = x - f0
        p = t * WT_SIZE
        i0 = int(p)
        if i0 >= WT_SIZE:
            i0 = WT_SIZE - 1
        pf = p - i0
        i1 = (i0 + 1) & (WT_SIZE - 1)
        a = wt[f0, i0] + pf * (wt[f0, i1] - wt[f0, i0])
        b = wt[f0 + 1, i0] + pf * (wt[f0 + 1, i1] - wt[f0 + 1, i0])
        return a + fr * (b - a)


@njit(cache=True)
def _lfo_wave(shape, t, sh):
    if shape == 0:
        return math.sin(2.0 * math.pi * t)
    elif shape == 1:
        return 1.0 - 4.0 * abs(t - 0.5)
    elif shape == 2:
        return 2.0 * t - 1.0
    elif shape == 3:
        return 1.0 if t < 0.5 else -1.0
    return sh


@njit(cache=True)
def _env_step(stage, level, a_inc, d_coef, sus, r_coef):
    if stage == 0:
        level += a_inc
        if level >= 1.0:
            level = 1.0
            stage = 1
    elif stage == 1:
        level = sus + (level - sus) * d_coef
    elif stage == 3:
        level *= r_coef
    return stage, level


@njit(cache=True)
def render_voice(p, n, gate, sr, wt):
    out = np.zeros(n, dtype=np.float32)
    w1 = _cat(p[0], 6); wtp1 = p[1]; lvl1 = p[2]
    semi1 = math.floor(-24.0 + 48.0 * p[3] + 0.5); cent1 = -50.0 + 100.0 * p[4]
    w2 = _cat(p[5], 6); wtp2 = p[6]; lvl2 = p[7]
    semi2 = math.floor(-24.0 + 48.0 * p[8] + 0.5); cent2 = -50.0 + 100.0 * p[9]
    fm = p[10] * p[10] * 2.0
    ring = p[11]
    sync = _cat(p[12], 2)
    midi = 24.0 + 72.0 * p[13]
    f0 = 440.0 * 2.0 ** ((midi - 69.0) / 12.0)
    ftype = _cat(p[14], 3)
    cutoff = 20.0 * 1000.0 ** p[15]
    k = 2.0 - 1.96 * p[16]
    env_oct = (2.0 * p[17] - 1.0) * 6.0
    fa = 1.0 / (_env_time(p[18]) * sr); fd = math.exp(-5.0 / (_env_time(p[19]) * sr))
    fs = p[20]; fr_ = math.exp(-5.0 / (_env_time(p[21]) * sr))
    aa = 1.0 / (_env_time(p[22]) * sr); ad = math.exp(-5.0 / (_env_time(p[23]) * sr))
    as_ = p[24]; ar = math.exp(-5.0 / (_env_time(p[25]) * sr))
    lrate = 0.05 * 400.0 ** p[27]
    lshape = _cat(p[28], 5)
    l_pitch = p[29] ** 3 * 12.0
    l_cut = p[30] * p[30] * 3.0
    l_amp = p[31]
    l_wt = p[32]

    f1b = f0 * 2.0 ** ((semi1 + cent1 / 100.0) / 12.0)
    f2b = f0 * 2.0 ** ((semi2 + cent2 / 100.0) / 12.0)

    rng_noise = 12345
    rng_lfo = 777
    rng_lfo, sh = _rand(rng_lfo)
    ph1 = 0.0; ph2 = 0.0; lph = 0.0
    fst = 0; flev = 0.0; ast = 0; alev = 0.0
    ic1 = 0.0; ic2 = 0.0; a1 = 0.0; a2 = 0.0; a3 = 0.0
    nyq = 0.45 * sr

    for i in range(n):
        if i == gate:
            fst = 3
            ast = 3
        fst, flev = _env_step(fst, flev, fa, fd, fs, fr_)
        ast, alev = _env_step(ast, alev, aa, ad, as_, ar)

        lph += lrate / sr
        if lph >= 1.0:
            lph -= 1.0
            if lshape == 4:
                rng_lfo, sh = _rand(rng_lfo)
        l = _lfo_wave(lshape, lph, sh)

        pm = 2.0 ** (l_pitch * l / 12.0)
        inc1 = min(f1b * pm / sr, 0.5)
        inc2 = min(f2b * pm / sr, 0.5)
        wp1 = min(max(wtp1 + l_wt * 0.5 * l, 0.0), 1.0)
        wp2 = min(max(wtp2 + l_wt * 0.5 * l, 0.0), 1.0)

        n2 = 0.0
        if w2 == 4:
            rng_noise, n2 = _rand(rng_noise)
        n1 = 0.0
        if w1 == 4:
            rng_noise, n1 = _rand(rng_noise)

        ph2 += inc2
        wrapped = False
        if ph2 >= 1.0:
            ph2 -= math.floor(ph2)
            wrapped = True
        o2 = _osc(w2, ph2, inc2, wp2, n2, wt)

        ph1 += inc1
        if ph1 >= 1.0:
            ph1 -= math.floor(ph1)
        if sync == 1 and wrapped:
            ph1 = ph2 * inc1 / inc2
            ph1 -= math.floor(ph1)
        pmod = ph1 + fm * o2
        pmod -= math.floor(pmod)
        o1 = _osc(w1, pmod, inc1, wp1, n1, wt)

        x = (1.0 - ring) * (lvl1 * o1 + lvl2 * o2) + ring * o1 * o2

        if i % CTRL_RATE == 0:
            fc = cutoff * 2.0 ** (env_oct * flev + l_cut * l)
            fc = min(max(fc, 20.0), nyq)
            g = math.tan(math.pi * fc / sr)
            a1 = 1.0 / (1.0 + g * (g + k))
            a2 = g * a1
            a3 = g * a2
        v3 = x - ic2
        v1 = a1 * ic1 + a2 * v3
        v2 = ic2 + a2 * ic1 + a3 * v3
        ic1 = 2.0 * v1 - ic1
        ic2 = 2.0 * v2 - ic2
        if ftype == 0:
            y = v2
        elif ftype == 1:
            y = v1
        else:
            y = x - k * v1 - v2

        trem = 1.0 - l_amp * (0.5 - 0.5 * l)
        out[i] = y * alev * trem
    return out


COMB_LENS = np.array([1116, 1188, 1277, 1356])
AP_LENS = np.array([556, 441])


@njit(cache=True)
def render_fx(x, p, sr):
    y = x.copy()
    n = x.shape[0]
    drive = 1.0 + 29.0 * p[33] * p[33]
    dmix = p[34]
    crate = 0.1 * 50.0 ** p[35]; cdepth = p[36]; cmix = p[37]
    rsize = p[38]; rdamp = p[39] * 0.4; rmix = p[40]

    if dmix > 0.0:
        for i in range(n):
            y[i] = (1.0 - dmix) * y[i] + dmix * math.tanh(drive * y[i])

    if cmix > 0.0:
        blen = int(0.03 * sr) + 2
        buf = np.zeros(blen)
        w = 0
        cph = 0.0
        for i in range(n):
            buf[w] = y[i]
            d = (0.007 + cdepth * 0.005 * (0.5 + 0.5 * math.sin(2.0 * math.pi * cph))) * sr
            cph += crate / sr
            if cph >= 1.0:
                cph -= 1.0
            rp = w - d
            if rp < 0.0:
                rp += blen
            i0 = int(rp)
            fr = rp - i0
            i1 = i0 + 1
            if i1 >= blen:
                i1 = 0
            dl = buf[i0] + fr * (buf[i1] - buf[i0])
            y[i] = (1.0 - cmix) * y[i] + cmix * 0.5 * (y[i] + dl)
            w += 1
            if w >= blen:
                w = 0

    if rmix > 0.0:
        scale = sr / 44100.0
        fb = 0.7 + 0.28 * rsize
        cl = np.empty(4, dtype=np.int64)
        for j in range(4):
            cl[j] = max(int(COMB_LENS[j] * scale), 1)
        al = np.empty(2, dtype=np.int64)
        for j in range(2):
            al[j] = max(int(AP_LENS[j] * scale), 1)
        cbuf = np.zeros((4, cl.max()))
        cidx = np.zeros(4, dtype=np.int64)
        cfilt = np.zeros(4)
        abuf = np.zeros((2, al.max()))
        aidx = np.zeros(2, dtype=np.int64)
        for i in range(n):
            inp = y[i] * 0.03
            acc = 0.0
            for j in range(4):
                o = cbuf[j, cidx[j]]
                cfilt[j] = o * (1.0 - rdamp) + cfilt[j] * rdamp
                cbuf[j, cidx[j]] = inp + cfilt[j] * fb
                cidx[j] += 1
                if cidx[j] >= cl[j]:
                    cidx[j] = 0
                acc += o
            for j in range(2):
                bo = abuf[j, aidx[j]]
                o = -acc + bo
                abuf[j, aidx[j]] = acc + bo * 0.5
                aidx[j] += 1
                if aidx[j] >= al[j]:
                    aidx[j] = 0
                acc = o
            y[i] = (1.0 - rmix) * y[i] + rmix * acc * 3.0
    return y


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def gate_samples(p: np.ndarray, n: int) -> int:
    return int(n * (0.05 + 0.95 * float(p[26])))


def render(p: np.ndarray, duration: float, sr: int = 22050,
           normalize: bool = True) -> np.ndarray:
    """Render one note of `duration` seconds from a normalised parameter vector."""
    p = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
    assert p.shape == (N_PARAMS,)
    n = max(int(duration * sr), 1)
    x = render_voice(p, n, gate_samples(p, n), float(sr), WAVETABLES)
    x = render_fx(x, p, float(sr)).astype(np.float32)
    if normalize:
        x = peak_normalize(x)
    return x


def render_gated(p: np.ndarray, n: int, gate: int, sr: int = 22050,
                 normalize: bool = True) -> np.ndarray:
    """Render exactly n samples with an explicit note-off sample (used by the
    finetuner when the comparison window is shorter than the full note)."""
    p = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
    x = render_voice(p, max(int(n), 1), int(gate), float(sr), WAVETABLES)
    x = render_fx(x, p, float(sr)).astype(np.float32)
    return peak_normalize(x) if normalize else x


def peak_normalize(x: np.ndarray, target: float = 0.9) -> np.ndarray:
    pk = float(np.max(np.abs(x))) if x.size else 0.0
    if pk < 1e-6 or not np.isfinite(pk):
        return np.zeros_like(x)
    return (x * (target / pk)).astype(np.float32)
