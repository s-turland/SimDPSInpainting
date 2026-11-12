"""Common interface to pretrained diffusion models.

SimDPS needs three things from a model: a denoiser in the variance-exploding
parameterisation of Karras et al. (2022), ``x = x0 + sigma * n``, and, for
latent models, the autoencoder's ``encode`` and ``decode``. Models trained
under another parameterisation are converted by the adapters below.
"""

import torch


class Prior(torch.nn.Module):
    """Base class for diffusion priors.

    The adapters below take ``encode``, ``decode`` and any of these attributes
    as keyword arguments.

    Attributes
    ----------
    sample_rate : int
        Audio sample rate the model was trained at.
    hop_length : int
        Audio samples per frame of the diffused signal (1 for waveform models).
    length : int
        Frames per training example; inpainting is done in windows of this size.
    pad_frames : int
        Extra frames appended during sampling and dropped afterwards.
    sigma_min, sigma_max : float
        Default noise range of the sampler.
    sigmas : list of float or None
        Fixed noise levels, for distilled models that need them.
    cache_id : str or None
        Identifies the autoencoder in the on-disk cache of encoded tracks. With
        ``None``, encodings are only kept in memory.
    """

    sample_rate = None
    hop_length = 1
    length = None
    pad_frames = 0
    sigma_min = 1e-3
    sigma_max = 20.0
    sigmas = None
    cache_id = None

    def denoise(self, x, sigma):
        """Estimate ``E[x0 | x]`` at noise level ``sigma`` of shape ``(batch,)``."""
        raise NotImplementedError

    def encode(self, audio):
        """Map ``(batch, channels, samples)`` audio to the diffused signal."""
        return audio

    def decode(self, x):
        return x


class _Wrapped(Prior):
    """Prior around a network, with optional autoencoder functions."""

    def __init__(self, net, sample_rate, length, encode=None, decode=None, **attrs):
        super().__init__()
        self.net = net
        self.sample_rate = sample_rate
        self.length = length
        self._encode = encode
        self._decode = decode
        for k, v in attrs.items():
            if not hasattr(Prior, k):
                raise TypeError(f"unexpected argument '{k}'")
            setattr(self, k, v)

    def encode(self, audio):
        return audio if self._encode is None else self._encode(audio)

    def decode(self, x):
        return x if self._decode is None else self._decode(x)


class EDMPrior(_Wrapped):
    """Network trained with the EDM preconditioning.

    ``net(c_in * x, c_noise)`` predicts ``F`` with
    ``x0 = c_skip * x + c_out * F`` and ``c_noise = log(sigma) / 4``.
    """

    def __init__(self, net, sigma_data, sample_rate, length, **kwargs):
        super().__init__(net, sample_rate, length, **kwargs)
        self.sigma_data = sigma_data

    def denoise(self, x, sigma):
        s = sigma.view(-1, *([1] * (x.ndim - 1)))
        sd = self.sigma_data
        c_skip = sd**2 / (s**2 + sd**2)
        c_out = s * sd / (s**2 + sd**2).sqrt()
        c_in = 1 / (s**2 + sd**2).sqrt()
        c_noise = (sigma.log() / 4).view(-1, 1)
        return c_skip * x + c_out * self.net(c_in * x, c_noise)


class FlowPrior(_Wrapped):
    """Network trained with rectified flow, ``x_t = t n + (1 - t) x0``.

    ``net(x_t, t)`` predicts the velocity ``n - x0``. A VE state at noise level
    ``sigma`` corresponds to ``t = sigma / (sigma + sigma_data)`` after scaling
    by ``(1 - t) / sigma_data`` (Sec. 4.1 of the paper, with ``sigma_data = 1``).
    """

    def __init__(self, net, sample_rate, length, sigma_data=1.0, **kwargs):
        super().__init__(net, sample_rate, length, **kwargs)
        self.sigma_data = sigma_data

    def denoise(self, x, sigma):
        s = sigma.view(-1, *([1] * (x.ndim - 1)))
        t = s / (s + self.sigma_data)
        xt = x * (1 - t) / self.sigma_data
        v = self.net(xt, t.view(-1))
        return (xt - v * t) * self.sigma_data


class DenoiserPrior(_Wrapped):
    """Wraps a function ``denoise(x, sigma)`` that is already a VE denoiser.

    Examples
    --------
    >>> prior = DenoiserPrior(my_denoiser, sample_rate=44100, length=2**18)
    """

    def denoise(self, x, sigma):
        return self.net(x, sigma)
