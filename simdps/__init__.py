"""Similarity-guided diffusion posterior sampling (SimDPS)."""

import torch

from . import guidance, priors, search
from .audio import load_audio, save_audio
from .config import load_config
from .inpainter import Checkpoint, Inpainter, Result
from .sampler import EDMSampler

__version__ = "0.1.0"
__all__ = ["Checkpoint", "EDMSampler", "Inpainter", "Result", "guidance", "load", "load_audio", "load_config",
           "priors", "save_audio", "search"]


def load(name="acestep", device=None, dtype=None, config=None, **kwargs):
    """Load a pretrained prior into an :class:`Inpainter`.

    The weights are downloaded on first use.

    Parameters
    ----------
    name : str, dict or list
        ``"acestep"``: ACE-Step 1.5, general music at 48 kHz.
        ``"acestep-turbo"``: its distilled version, sampled in 7 steps.
        ``"cqtdiff"``: the piano model of the paper, from its checkpoint.
        Otherwise a config with a ``prior`` section, e.g.
        ``"ismir26_piano_strict"``, as taken by :func:`load_config`.
    device : str, optional
        Defaults to CUDA, then MPS, then CPU.
    dtype : torch.dtype, optional
        Defaults to bfloat16 on GPU and float32 on CPU.
    config : optional
        Inpainter settings, applied on top of any in ``name``.
    **kwargs
        Passed to the prior, e.g. ``checkpoint``.
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    if dtype is None:
        dtype = torch.float32 if device == "cpu" else torch.bfloat16
    if isinstance(name, str) and name in priors.PRIORS:
        settings, base = {"name": name}, None
    else:
        base = load_config(name)
        if "prior" not in base:
            raise ValueError(f"{name} does not say which model to load")
        settings = base["prior"]
    prior = priors.build_prior({**settings, **kwargs}, device=device, dtype=dtype)
    return Inpainter(prior, config=[base, config])
