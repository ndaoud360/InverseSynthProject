"""
Front-end: audio loading, log-mel features, onset detection and the dynamic
5-second window selector.

Everything here is implemented in plain NumPy (no librosa at runtime) so the
exact same maths can be reproduced in cpp/core/Features.cpp. The mel filterbank
follows librosa's Slaney definition, so librosa.feature.melspectrogram with the
same settings gives the same numbers (see tests/test_features.py).
"""
from dataclasses import dataclass
import math
import numpy as np

SR = 22050
N_FFT = 1024
HOP = 256
N_MELS = 128
FMIN = 20.0
FMAX = SR / 2
WINDOW_SEC = 5.0          # optimal analysis length for the predictor
MIN_SEC = 0.25
DB_FLOOR = -100.0
FRAMES_PER_SEC = SR / HOP  # ~86.13


# ----------------------------------------------------------------- mel basis
def _hz_to_mel(f):
    f = np.asarray(f, dtype=np.float64)
    f_sp = 200.0 / 3
    mels = f / f_sp
    min_log_hz, min_log_mel, logstep = 1000.0, 1000.0 / f_sp, math.log(6.4) / 27.0
    return np.where(f >= min_log_hz,
                    min_log_mel + np.log(np.maximum(f, 1e-10) / min_log_hz) / logstep, mels)


def _mel_to_hz(m):
    m = np.asarray(m, dtype=np.float64)
    f_sp = 200.0 / 3
    freqs = f_sp * m
    min_log_hz, min_log_mel, logstep = 1000.0, 1000.0 / f_sp, math.log(6.4) / 27.0
    return np.where(m >= min_log_mel, min_log_hz * np.exp(logstep * (m - min_log_mel)), freqs)


def mel_filterbank(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX) -> np.ndarray:
    fft_f = np.linspace(0, sr / 2, 1 + n_fft // 2)
    mel_f = _mel_to_hz(np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2))
    fdiff = np.diff(mel_f)
    ramps = mel_f[:, None] - fft_f[None, :]
    w = np.zeros((n_mels, len(fft_f)))
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        w[i] = np.maximum(0, np.minimum(lower, upper))
    w *= (2.0 / (mel_f[2:n_mels + 2] - mel_f[:n_mels]))[:, None]
    return w.astype(np.float32)


MEL_BASIS = mel_filterbank()
HANN = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)).astype(np.float32)


# ----------------------------------------------------------------- features
def power_stft(x: np.ndarray, n_fft=N_FFT, hop=HOP) -> np.ndarray:
    """Centred (zero-padded) power STFT. Returns (1 + n_fft/2, frames)."""
    x = np.pad(x.astype(np.float32), (n_fft // 2, n_fft // 2))
    n_frames = 1 + (len(x) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    win = HANN if n_fft == N_FFT else np.hanning(n_fft + 1)[:-1].astype(np.float32)
    spec = np.fft.rfft(x[idx] * win, axis=1)
    return (spec.real ** 2 + spec.imag ** 2).T.astype(np.float32)


def mel_db(x: np.ndarray) -> np.ndarray:
    mel = MEL_BASIS @ power_stft(x)
    return 10.0 * np.log10(np.maximum(mel, 1e-10))


def db_to_feature(db: np.ndarray) -> np.ndarray:
    """Maps dB (on peak-normalised audio) to roughly [-1, 1]."""
    return ((np.maximum(db, DB_FLOOR) + 50.0) / 50.0).astype(np.float32)


def log_mel(x: np.ndarray) -> np.ndarray:
    """(N_MELS, T) network input for a peak-normalised signal."""
    return db_to_feature(mel_db(x))


# ------------------------------------------------------ onset / window select
def _z(v):
    """z-score clipped to +-3 so a single transient cannot dominate a window."""
    s = v.std()
    return np.clip((v - v.mean()) / (s + 1e-8), -3.0, 3.0)


def onset_envelope(db: np.ndarray) -> np.ndarray:
    """Half-wave rectified spectral flux on the log-mel spectrogram."""
    d = np.maximum(db, DB_FLOOR)
    prev = np.concatenate([np.full((d.shape[0], 1), DB_FLOOR), d[:, :-1]], axis=1)
    return np.maximum(d - prev, 0.0).mean(axis=0)


def pick_onsets(env: np.ndarray, k: float = 1.5, min_gap_frames: int = 4) -> np.ndarray:
    thr = env.mean() + k * env.std()
    peaks, last = [], -10 ** 9
    for t in range(len(env)):
        l = env[t - 1] if t > 0 else -np.inf
        r = env[t + 1] if t + 1 < len(env) else -np.inf
        if env[t] >= thr and env[t] >= l and env[t] > r and t - last >= min_gap_frames:
            peaks.append(t)
            last = t
    return np.array(peaks, dtype=np.int64)


def density_curve(db: np.ndarray) -> np.ndarray:
    """
    Per-frame 'sonic density' = transient activity + loudness + spectral
    complexity (entropy of the mel power distribution).
    """
    flux = onset_envelope(db)
    p = 10.0 ** (np.maximum(db, DB_FLOOR) / 10.0)
    energy = 10.0 * np.log10(p.sum(axis=0) + 1e-10)          # frame loudness (dB)
    p = p / (p.sum(axis=0, keepdims=True) + 1e-12)
    entropy = -(p * np.log(p + 1e-12)).sum(axis=0) / math.log(db.shape[0])
    return _z(flux) + _z(energy) + 0.5 * _z(entropy)


@dataclass
class Window:
    start: int           # samples
    length: int          # samples
    total: int           # samples

    @property
    def ctx(self) -> np.ndarray:
        """Context vector given to the predictor alongside the spectrogram."""
        total_s = self.total / SR
        return np.array([self.start / max(self.total, 1),
                         (self.length / SR) / WINDOW_SEC,
                         min(total_s, 60.0) / 60.0], dtype=np.float32)


def select_window(x: np.ndarray, window_sec: float = WINDOW_SEC,
                  snap_sec: float = 2.0) -> Window:
    """
    Pick the most sonically dense `window_sec` region. Files at or below the
    window length are used whole. The chosen start is snapped back to the
    nearest onset within `snap_sec` so the note attack is inside the window.
    """
    n = len(x)
    wlen = int(window_sec * SR)
    if n <= wlen:
        return Window(0, n, n)
    # analysis at a coarser hop keeps this O(n) and cheap even for long files
    db = mel_db(x)
    dens = density_curve(db)
    wf = int(round(window_sec * FRAMES_PER_SEC))
    cs = np.concatenate([[0.0], np.cumsum(dens)])
    scores = cs[wf:] - cs[:-wf]
    best = int(np.argmax(scores))
    onsets = pick_onsets(onset_envelope(db))
    snap = int(snap_sec * FRAMES_PER_SEC)
    cand = onsets[(onsets <= best) & (onsets >= best - snap)]
    if len(cand):
        best = int(cand[0])
    start = min(best * HOP, n - wlen)
    return Window(start, wlen, n)


def prepare_input(x: np.ndarray, window_sec: float = WINDOW_SEC):
    """Full pre-processing for one audio clip -> (mel (128,T), ctx (3,), Window)."""
    x = peak_normalize_f(x)
    w = select_window(x, window_sec)
    seg = peak_normalize_f(x[w.start:w.start + w.length])
    return log_mel(seg), w.ctx, w


def peak_normalize_f(x, target=0.9):
    pk = float(np.max(np.abs(x))) if len(x) else 0.0
    return x if pk < 1e-6 else (x * (target / pk)).astype(np.float32)


# ------------------------------------------------------------------ I/O
def load_audio(path: str, sr: int = SR) -> np.ndarray:
    """Load any-length file, mix to mono, resample to `sr`."""
    import soundfile as sf
    from scipy.signal import resample_poly
    x, file_sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if file_sr != sr:
        g = math.gcd(int(file_sr), sr)
        x = resample_poly(x, sr // g, int(file_sr) // g).astype(np.float32)
    # trim leading silence (below -60 dB of peak) so onsets align with note-on
    pk = np.max(np.abs(x)) + 1e-12
    nz = np.nonzero(np.abs(x) > pk * 1e-3)[0]
    if len(nz):
        x = x[max(nz[0] - 32, 0):]
    if len(x) < int(MIN_SEC * sr):
        x = np.pad(x, (0, int(MIN_SEC * sr) - len(x)))
    return x
