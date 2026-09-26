"""
Single source of truth for the synth parameter vector.

Every parameter is stored normalised to [0, 1]. The mapping from normalised
value -> physical value lives in `engine.py` (Python) and `core/Params.h` (C++)
and MUST be kept identical. `tests/test_parity.py` checks this.

N_PARAMS = 41
"""
from dataclasses import dataclass
import numpy as np

WAVE_NAMES = ["sine", "saw", "square", "triangle", "noise", "wavetable"]
FILTER_NAMES = ["lowpass", "bandpass", "highpass"]
LFO_SHAPES = ["sine", "triangle", "saw", "square", "sample&hold"]


@dataclass(frozen=True)
class ParamSpec:
    name: str
    n_classes: int = 0          # 0 = continuous, >0 = categorical
    default: float = 0.5

    @property
    def categorical(self) -> bool:
        return self.n_classes > 0


PARAMS = [
    # --- Oscillator 1 (carrier / sync slave) -------------------------------
    ParamSpec("osc1_wave", 6, 1 / 12 + 1 / 6),   # saw
    ParamSpec("osc1_wtpos", 0, 0.0),
    ParamSpec("osc1_level", 0, 0.8),
    ParamSpec("osc1_coarse", 0, 0.5),            # -24..+24 semitones
    ParamSpec("osc1_fine", 0, 0.5),              # -50..+50 cents
    # --- Oscillator 2 (modulator / sync master) ----------------------------
    ParamSpec("osc2_wave", 6, 1 / 12),           # sine
    ParamSpec("osc2_wtpos", 0, 0.0),
    ParamSpec("osc2_level", 0, 0.0),
    ParamSpec("osc2_coarse", 0, 0.5),
    ParamSpec("osc2_fine", 0, 0.5),
    # --- Cross modulation ----------------------------------------------------
    ParamSpec("fm_amount", 0, 0.0),
    ParamSpec("ring_mix", 0, 0.0),
    ParamSpec("sync", 2, 0.25),                  # off
    # --- Pitch -------------------------------------------------------------
    ParamSpec("pitch", 0, 0.5),                  # MIDI 24..96
    # --- Filter ------------------------------------------------------------
    ParamSpec("filt_type", 3, 1 / 6),            # lowpass
    ParamSpec("filt_cutoff", 0, 0.8),
    ParamSpec("filt_res", 0, 0.1),
    ParamSpec("filt_env_amt", 0, 0.5),           # bipolar, 0.5 = none
    ParamSpec("fenv_attack", 0, 0.1),
    ParamSpec("fenv_decay", 0, 0.5),
    ParamSpec("fenv_sustain", 0, 0.5),
    ParamSpec("fenv_release", 0, 0.5),
    # --- Amp envelope ------------------------------------------------------
    ParamSpec("aenv_attack", 0, 0.1),
    ParamSpec("aenv_decay", 0, 0.5),
    ParamSpec("aenv_sustain", 0, 0.8),
    ParamSpec("aenv_release", 0, 0.4),
    ParamSpec("gate", 0, 0.8),                   # held fraction of render length
    # --- LFO (multi-routing: one depth per destination) --------------------
    ParamSpec("lfo_rate", 0, 0.5),
    ParamSpec("lfo_shape", 5, 0.1),
    ParamSpec("lfo_to_pitch", 0, 0.0),
    ParamSpec("lfo_to_cutoff", 0, 0.0),
    ParamSpec("lfo_to_amp", 0, 0.0),
    ParamSpec("lfo_to_wtpos", 0, 0.0),
    # --- FX ----------------------------------------------------------------
    ParamSpec("dist_drive", 0, 0.3),
    ParamSpec("dist_mix", 0, 0.0),
    ParamSpec("chorus_rate", 0, 0.3),
    ParamSpec("chorus_depth", 0, 0.5),
    ParamSpec("chorus_mix", 0, 0.0),
    ParamSpec("rev_size", 0, 0.5),
    ParamSpec("rev_damp", 0, 0.5),
    ParamSpec("rev_mix", 0, 0.0),
]

N_PARAMS = len(PARAMS)
assert N_PARAMS == 41
IDX = {p.name: i for i, p in enumerate(PARAMS)}
CAT_IDX = [i for i, p in enumerate(PARAMS) if p.categorical]
CAT_CLASSES = [PARAMS[i].n_classes for i in CAT_IDX]
CONT_IDX = [i for i, p in enumerate(PARAMS) if not p.categorical]
DEFAULTS = np.array([p.default for p in PARAMS], dtype=np.float32)


def cat_to_norm(k: int, n: int) -> float:
    """Class index -> normalised value (bin centre)."""
    return (k + 0.5) / n


def norm_to_cat(v: float, n: int) -> int:
    return int(min(max(int(v * n), 0), n - 1))


def relevance_mask(y: np.ndarray) -> np.ndarray:
    """
    Which parameters actually influence the sound for a given vector.
    Irrelevant parameters (e.g. wavetable position when the osc is a saw)
    are un-learnable and must be excluded from the parameter loss.
    y: (..., N) normalised. Returns float mask of same shape.
    """
    m = np.ones_like(y, dtype=np.float32)
    w1 = np.minimum((y[..., IDX["osc1_wave"]] * 6).astype(int), 5)
    w2 = np.minimum((y[..., IDX["osc2_wave"]] * 6).astype(int), 5)
    m[..., IDX["osc1_wtpos"]] = (w1 == 5)
    osc2_used = ((y[..., IDX["osc2_level"]] > 0.02) | (y[..., IDX["fm_amount"]] > 0.02)
                 | (y[..., IDX["ring_mix"]] > 0.02) | (y[..., IDX["sync"]] >= 0.5))
    for n in ["osc2_wave", "osc2_coarse", "osc2_fine"]:
        m[..., IDX[n]] = osc2_used
    m[..., IDX["osc2_wtpos"]] = osc2_used & (w2 == 5)
    fenv_used = np.abs(y[..., IDX["filt_env_amt"]] - 0.5) > 0.02
    for n in ["fenv_attack", "fenv_decay", "fenv_sustain", "fenv_release"]:
        m[..., IDX[n]] = fenv_used
    lfo_used = np.zeros(y.shape[:-1], dtype=bool)
    for n in ["lfo_to_pitch", "lfo_to_cutoff", "lfo_to_amp", "lfo_to_wtpos"]:
        lfo_used |= y[..., IDX[n]] > 0.02
    m[..., IDX["lfo_rate"]] = lfo_used
    m[..., IDX["lfo_shape"]] = lfo_used
    m[..., IDX["dist_drive"]] = y[..., IDX["dist_mix"]] > 0.02
    for n in ["chorus_rate", "chorus_depth"]:
        m[..., IDX[n]] = y[..., IDX["chorus_mix"]] > 0.02
    for n in ["rev_size", "rev_damp"]:
        m[..., IDX[n]] = y[..., IDX["rev_mix"]] > 0.02
    return m


def random_params(rng: np.random.Generator) -> np.ndarray:
    """
    Musically-biased random patch. Uniform sampling over the cube produces
    mostly noise; these priors make the dataset look like real synth sounds.
    """
    y = rng.random(N_PARAMS).astype(np.float32)

    def snap_semis(p_zero):
        r = rng.random()
        if r < p_zero:
            s = 0
        elif r < p_zero + 0.25:
            s = rng.choice([-12, 12, 7, -5, 24, -24, 19])
        else:
            s = rng.integers(-24, 25)
        return (s + 24) / 48.0

    y[IDX["osc1_coarse"]] = snap_semis(0.85)
    y[IDX["osc2_coarse"]] = snap_semis(0.35)
    for n in ["osc1_fine", "osc2_fine"]:
        y[IDX[n]] = np.clip(0.5 + rng.normal(0, 0.08), 0, 1) if rng.random() < 0.5 else 0.5
    y[IDX["osc1_level"]] = rng.uniform(0.3, 1.0)
    y[IDX["osc2_level"]] = 0.0 if rng.random() < 0.4 else rng.uniform(0, 1)
    y[IDX["fm_amount"]] = 0.0 if rng.random() < 0.6 else rng.random()
    y[IDX["ring_mix"]] = 0.0 if rng.random() < 0.8 else rng.random()
    y[IDX["sync"]] = cat_to_norm(int(rng.random() < 0.15), 2)
    for n in ["osc1_wave", "osc2_wave"]:
        y[IDX[n]] = cat_to_norm(rng.choice(6, p=[.2, .25, .2, .12, .05, .18]), 6)
    y[IDX["filt_type"]] = cat_to_norm(rng.choice(3, p=[.7, .15, .15]), 3)
    y[IDX["filt_env_amt"]] = 0.5 if rng.random() < 0.3 else rng.random()
    y[IDX["lfo_shape"]] = cat_to_norm(rng.integers(5), 5)
    for n in ["lfo_to_pitch", "lfo_to_cutoff", "lfo_to_amp", "lfo_to_wtpos"]:
        y[IDX[n]] = 0.0 if rng.random() < 0.7 else rng.random()
    for n in ["dist_mix", "chorus_mix", "rev_mix"]:
        y[IDX[n]] = 0.0 if rng.random() < 0.5 else rng.random()
    y[IDX["pitch"]] = rng.beta(2.2, 2.2)          # favour mid register
    y[IDX["gate"]] = rng.uniform(0.2, 1.0)
    return y
