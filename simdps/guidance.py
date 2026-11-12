"""Likelihood guidance for posterior sampling.

``Observation`` is the DPS term of Chung et al. (2023), Eq. (5) of the paper, and
``Candidates`` the auxiliary term of Eq. (8). Each gradient is rescaled to an RMS
equal to its weight.
"""

import math

import torch

from .sampler import _expand


def gap_mask(length, start, end, channels=1, device=None):
    """Binary mask with zeros on frames ``[start, end)``."""
    m = torch.ones(channels, length, device=device)
    m[:, start:end] = 0.0
    return m


class Guidance:
    """Adds the gradients of a list of terms to the ODE direction.

    Parameters
    ----------
    terms : list
        Objects with ``weight(sigma)`` and ``loss(x0_hat)``. With ``project``,
        the order sets the priority: a later term is projected away from the
        terms before it when they conflict.
    project : bool
        Remove conflicting components, first between terms, then between the
        guidance and the unconditional direction.
    length : int, optional
        Frames passed to the terms. Extra frames, which some priors append,
        are generated freely.
    batched : bool
        Backpropagate all terms through the denoiser in one batched pass
        rather than one pass each. Falls back to separate passes if the
        denoiser does not support it.
    jacobian : bool
        Differentiate through the denoiser. With ``False`` the gradients are
        taken with respect to the denoised estimate instead, which needs no
        backward pass through the network.
    joint : bool
        Take one gradient of the first loss plus the others times their
        weights, rescaled to an RMS equal to the first term's weight, instead
        of rescaling each term's gradient.
    """

    def __init__(self, terms, project=True, length=None, batched=True, jacobian=True, joint=False):
        self.terms = list(terms)
        self.project = project
        self.length = length
        self.batched = batched
        self.jacobian = jacobian
        self.joint = joint

    def __call__(self, denoise, x, sigma):
        active = [(t, t.weight(sigma)) for t in self.terms]
        active = [(t, w) for t, w in active if w is not None]
        through = self.jacobian and bool(active)
        with torch.set_grad_enabled(bool(active)):
            x = x.detach().requires_grad_(through)
            with torch.set_grad_enabled(through):
                x0 = denoise(x, sigma)
            d = ((x - x0) / _expand(sigma, x)).detach()
            if not through:
                x0 = x0.detach().requires_grad_(bool(active))
            x0_obs = x0[..., : self.length] if self.length else x0

            losses = [t.loss(x0_obs) for t, _ in active]
            if self.joint and active:
                total = losses[0] + sum(_mean_weight(w) * loss for (_, w), loss in zip(active[1:], losses[1:]))
                g = torch.autograd.grad(total, x if through else x0)[0]
                g = g * (_expand(_as_tensor(active[0][1], x), x) / (_rms(g) + 1e-8))
                if self.project:
                    g = _deconflict(g, d)
                return d + g, x0.detach()
            if through:
                grads = self._gradients(losses, x0, x)
            else:
                grads = [torch.autograd.grad(loss, x0, retain_graph=True)[0] for loss in losses]
            g = torch.zeros_like(x)
            for k, ((_, w), gk) in enumerate(zip(active, grads)):
                gk = gk * (_expand(_as_tensor(w, x), x) / (_rms(gk) + 1e-8))
                if self.project and k > 0:
                    gk = _deconflict(gk, g)
                g = g + gk
            if self.project:
                g = _deconflict(g, d)
        return d + g, x0.detach()

    def _gradients(self, losses, x0, x):
        if len(losses) < 2 or not self.batched:
            return [torch.autograd.grad(loss, x, retain_graph=True)[0] for loss in losses]
        # the losses are cheap functions of x0, so differentiate up to x0 first
        # and run one batched vector-Jacobian product through the denoiser
        v = torch.stack([torch.autograd.grad(loss, x0, retain_graph=True)[0] for loss in losses])
        try:
            return list(torch.autograd.grad(x0, x, v, retain_graph=True, is_grads_batched=True)[0])
        except RuntimeError:
            self.batched = False
            return [torch.autograd.grad(x0, x, vk, retain_graph=True)[0] for vk in v]


class Observation:
    """DPS term ``||M (y - x0_hat)||^2`` on the observed frames.

    Parameters
    ----------
    y : Tensor
        Observation, zero inside the gap.
    mask : Tensor
        1 on observed frames, 0 in the gap. Broadcasts against ``y``.
    zeta : float
        Guidance weight.
    loss : {"l2", "comp_stft"}
        Squared error of the signals, or of their compressed spectrograms
        (see :func:`compressed_stft`).
    compression : float
        Exponent of the magnitudes for ``"comp_stft"``.
    """

    def __init__(self, y, mask, zeta=0.5, loss="l2", compression=0.3):
        if loss not in ("l2", "comp_stft"):
            raise ValueError(f"unknown loss '{loss}'")
        self.y = y
        self.mask = mask
        self.zeta = zeta
        self.kind = loss
        self.compression = compression

    def weight(self, sigma):
        return self.zeta

    def loss(self, x0):
        if self.kind == "comp_stft":
            return _comp_stft_error(self.mask * x0, self.y, self.compression).sum()
        err = (self.mask * x0 - self.y) ** 2
        return err.flatten(1).mean(1).sum()


class Candidates:
    """SimDPS term pulling the gap towards the retrieved candidates.

    Parameters
    ----------
    candidates : Tensor
        ``(K, C, T)`` candidate signals, ranked by score. Only the gap is used.
    scores : Tensor
        ``(K,)`` similarity scores.
    gap : tuple of int
        ``(start, end)`` frames of the gap.
    weight : float
        Guidance weight.
    combine : {"weighted", "solo", "min", "mean"}
        ``"weighted"`` (SimDPS-Multi, Eq. 9) sums the distances weighted by the
        softmax of the scores; ``"solo"`` (SimDPS-Solo) uses the top candidate only.
    distance : {"lowpass", "magnitude", "l2", "comp_stft"}
        ``"lowpass"`` is the L1 distance between low-passed gaps, ``"magnitude"``
        the L2 distance between their low-passed magnitude spectra (Eq. 10),
        ``"l2"`` the plain squared error, and ``"comp_stft"`` the squared error
        of their compressed spectrograms.
    compression : float
        Exponent of the magnitudes for ``"comp_stft"``.
    cutoff : int
        First discarded frequency bin of the low-pass filter.
    taper : float or None
        Tukey taper over the gap before filtering, as a fraction of its length.
    stop : float
        Noise level at and below which the term is switched off.
    fade : float
        Exponent of the weight decay ``((sigma - lo) / (hi - lo)) ** fade``;
        0 keeps the weight constant until ``stop``.
    sigma_range : tuple of float
        ``(lo, hi)`` used by the fade, normally the sampler's schedule limits.
    weights : Tensor, optional
        ``(K,)`` shares of the ``"weighted"`` combination, used instead of the
        softmax of the scores.
    """

    def __init__(
        self,
        candidates,
        scores,
        gap,
        weight=0.1,
        combine="weighted",
        distance="lowpass",
        cutoff=8,
        taper=0.5,
        stop=0.5,
        fade=0.5,
        sigma_range=(1e-3, 20.0),
        weights=None,
        compression=0.3,
    ):
        if combine not in ("weighted", "solo", "min", "mean"):
            raise ValueError(f"unknown combine mode '{combine}'")
        if distance not in ("lowpass", "magnitude", "l2", "comp_stft"):
            raise ValueError(f"unknown distance '{distance}'")
        self.candidates = candidates
        self.scores = scores
        self.start, self.end = gap
        self.w = weight
        self.combine = combine
        self.distance = distance
        self.cutoff = cutoff
        self.taper = taper
        self.stop = stop
        self.fade = fade
        self.sigma_range = sigma_range
        self.weights = weights
        self.compression = compression

        self.mask = gap_mask(candidates.shape[-1], *gap, device=candidates.device)
        self.window = None
        if taper:
            self.window = tukey(self.end - self.start, taper, device=candidates.device)

    def weight(self, sigma):
        if float(sigma.max()) <= self.stop:
            return None
        lo, hi = self.sigma_range
        return self.w * ((sigma - lo) / (hi - lo + 1e-5)) ** self.fade

    def loss(self, x0):
        cands = self.candidates[:1] if self.combine == "solo" else self.candidates
        d = torch.stack([self.dist(x0, c) for c in cands])
        if self.combine == "weighted":
            w = (torch.softmax(self.scores, 0) if self.weights is None else self.weights).to(d)
            d = (w[:, None] * d).sum(0)
        elif self.combine == "min":
            d = d.min(0).values
        elif self.combine == "mean":
            d = d.mean(0)
        else:
            d = d[0]
        return d.sum()

    def dist(self, x0, cand):
        """Per-item distance between the gap of ``x0`` and of one candidate."""
        a = (1 - self.mask) * x0
        b = (1 - self.mask) * cand
        if self.distance == "l2":
            return ((a - b) ** 2).flatten(1).mean(1)
        if self.distance == "comp_stft":
            return _comp_stft_error(a, b.expand_as(a), self.compression)

        a = a[..., self.start : self.end]
        b = b[..., self.start : self.end].expand_as(a)
        if self.window is not None:
            a, b = a * self.window, b * self.window
        A = torch.fft.rfft(a, dim=-1)
        B = torch.fft.rfft(b, dim=-1)
        if self.distance == "magnitude":
            A, B = A[..., : self.cutoff], B[..., : self.cutoff]
            return ((A.abs() - B.abs()) ** 2).flatten(1).mean(1)
        A[..., self.cutoff :] = 0
        B[..., self.cutoff :] = 0
        n = a.shape[-1]
        a = torch.fft.irfft(A, n=n, dim=-1)
        b = torch.fft.irfft(B, n=n, dim=-1)
        return (a - b).abs().flatten(1).mean(1)


def tukey(n, alpha, device=None):
    """Tukey window with cosine tapers covering ``alpha / 2`` of each end."""
    w = torch.ones(n, device=device)
    k = max(1, int(n * alpha / 2))
    ramp = torch.linspace(0, 1, k, device=device)
    w[:k] = 0.5 * (1 - torch.cos(math.pi * ramp))
    w[-k:] = 0.5 * (1 + torch.cos(math.pi * ramp))
    return w


def compressed_stft(x, power=0.3, win_length=1024, n_fft=2048, hop_length=256):
    """Spectrogram of ``(..., samples)`` with its magnitudes raised to ``power``."""
    shape = x.shape
    window = torch.hann_window(win_length, device=x.device, dtype=x.dtype)
    X = torch.stft(x.reshape(-1, shape[-1]), n_fft, hop_length, win_length, window, center=True, return_complex=True)
    X = (X.abs() + 1e-8) ** power * torch.exp(1j * X.angle())
    return X.reshape(*shape[:-1], *X.shape[-2:])


def _comp_stft_error(a, b, power):
    # per-item error, like the other distances
    return (compressed_stft(a, power) - compressed_stft(b, power)).abs().pow(2).flatten(1).mean(1)


def _rms(g):
    return g.flatten(1).norm(dim=1).view(-1, *([1] * (g.ndim - 1))) / math.sqrt(g[0].numel())


def _mean_weight(w):
    return w.mean() if torch.is_tensor(w) else w


def _as_tensor(w, x):
    if torch.is_tensor(w):
        return w.to(x).expand(x.shape[0])
    return x.new_full((x.shape[0],), w)


def _deconflict(g, ref):
    """Project ``g`` onto the plane normal to ``ref`` where they point apart."""
    gf, rf = g.flatten(1), ref.flatten(1)
    dot = (gf * rf).sum(1, keepdim=True)
    proj = gf - dot / ((rf * rf).sum(1, keepdim=True) + 1e-8) * rf
    return torch.where(dot < 0, proj, gf).view_as(g)
