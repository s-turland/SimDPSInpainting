"""Audio file input and output."""

import math
import warnings

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly


def load_audio(path, sample_rate=None):
    """Read an audio file as a ``(channels, samples)`` float tensor.

    Returns the audio and its sample rate, resampled if ``sample_rate`` is given.
    """
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = torch.from_numpy(x.T.copy())
    if sample_rate is not None and sample_rate != sr:
        x, sr = resample(x, sr, sample_rate), sample_rate
    return x, sr


def save_audio(path, audio, sample_rate):
    """Write ``(channels, samples)`` audio. The format follows the extension.

    WAV files are written as 32-bit float, as generated audio can exceed full
    scale. Other formats are clipped, with a warning.
    """
    audio = torch.as_tensor(audio).detach().cpu().float()
    if audio.ndim == 1:
        audio = audio[None]
    subtype = "FLOAT" if str(path).lower().endswith(".wav") else None
    if subtype is None and audio.abs().max() > 1:
        warnings.warn(f"clipping {path}, peak {audio.abs().max():.2f}", stacklevel=2)
    sf.write(path, audio.numpy().T, sample_rate, subtype=subtype)


def resample(x, orig, new):
    """Polyphase resampling along the last axis."""
    if orig == new:
        return x
    g = math.gcd(int(orig), int(new))
    y = resample_poly(x.cpu().numpy(), int(new) // g, int(orig) // g, axis=-1)
    return torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32)).to(x.device)


def crossfade(a, b, start, end, fade):
    """``a`` outside ``[start, end)``, ``b`` inside, with equal-power fades of
    ``fade`` samples just outside the interval."""
    out = a.clone()
    out[..., start:end] = b[..., start:end]
    n = a.shape[-1]
    for lo, hi, rising in ((start - fade, start, True), (end, end + fade, False)):
        lo, hi = max(lo, 0), min(hi, n)
        if hi <= lo:
            continue
        t = torch.linspace(0, 1, hi - lo + 2, device=a.device)[1:-1]
        if not rising:
            t = 1 - t
        w_in, w_out = torch.sin(0.5 * math.pi * t), torch.cos(0.5 * math.pi * t)
        out[..., lo:hi] = w_out * a[..., lo:hi] + w_in * b[..., lo:hi]
    return out
