"""Stochastic Euler/Heun sampler of Karras et al. (2022), Algorithm 2."""

import torch


class EDMSampler:
    """Solves the probability-flow ODE from ``sigma_max`` down to zero.

    Parameters
    ----------
    steps : int
        Number of noise levels.
    sigma_min, sigma_max, rho : float
        Schedule ``sigma_i = (a + i / (steps - 1) * (b - a)) ** rho`` with
        ``a = sigma_max ** (1 / rho)`` and ``b = sigma_min ** (1 / rho)``.
    churn : float
        Amount of noise re-injected at each step (``S_churn``). 0 gives a
        deterministic sampler.
    churn_range : tuple of float
        Noise levels between which churn is applied.
    s_noise : float
        Scale of the re-injected noise.
    order : int
        1 for Euler, 2 for Heun.
    sigmas : list of float, optional
        Explicit decreasing noise levels, e.g. those a distilled model was
        trained on. Overrides ``steps`` and the schedule parameters.
    """

    def __init__(
        self,
        steps=100,
        sigma_min=1e-3,
        sigma_max=20.0,
        rho=7.0,
        churn=10.0,
        churn_range=(0.0, float("inf")),
        s_noise=1.0,
        order=1,
        sigmas=None,
    ):
        if order not in (1, 2):
            raise ValueError("order must be 1 or 2")
        self.fixed = None if sigmas is None else [float(s) for s in sigmas]
        if self.fixed:
            steps, sigma_max, sigma_min = len(self.fixed), self.fixed[0], self.fixed[-1]
        elif steps < 2:
            raise ValueError("the schedule needs at least 2 steps; use sigmas for a single level")
        self.steps = steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.churn = churn
        self.churn_range = churn_range
        self.s_noise = s_noise
        self.order = order

    def sigmas(self, device=None, dtype=torch.float32):
        """Noise levels, with a final zero appended."""
        if self.fixed:
            return torch.tensor(self.fixed + [0.0], device=device, dtype=dtype)
        i = torch.arange(self.steps, device=device, dtype=dtype)
        a = self.sigma_max ** (1 / self.rho)
        b = self.sigma_min ** (1 / self.rho)
        s = (a + i / (self.steps - 1) * (b - a)) ** self.rho
        return torch.cat([s, torch.zeros_like(s[:1])])

    def _gamma(self, sigma):
        lo, hi = self.churn_range
        if lo <= sigma <= hi:
            return min(self.churn / self.steps, 2**0.5 - 1)
        return 0.0

    def sample(self, denoise, x, guidance=None, generator=None, callback=None, start=0):
        """Draw a sample.

        Parameters
        ----------
        denoise : callable
            ``denoise(x, sigma) -> x0_hat``, with ``sigma`` of shape ``(batch,)``.
        x : Tensor
            Starting point at ``sigma_max``, e.g. ``sigma_max * randn``.
        guidance : Guidance, optional
            Adds likelihood terms to the ODE direction (see :mod:`simdps.guidance`).
        generator : torch.Generator, optional
            Source of the churn noise.
        callback : callable, optional
            Called as ``callback(i, sigma, x, x0_hat)`` after each step.
        start : int
            Step to begin at, with ``x`` at noise level ``sigmas()[start]``,
            e.g. to resume an earlier run from one of its steps.

        Returns
        -------
        Tensor
            Sample at ``sigma = 0``.
        """
        sigmas = self.sigmas(x.device, x.dtype)
        ones = x.new_ones(x.shape[0])

        for i in range(start, self.steps):
            s, s_next = sigmas[i], sigmas[i + 1]
            # increase noise temporarily
            gamma = self._gamma(s)
            s_hat = s + gamma * s
            if gamma > 0:
                eps = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
                x = x + (s_hat**2 - s**2).sqrt() * self.s_noise * eps

            # Euler step, then the second-order correction
            d, x0 = self._direction(denoise, guidance, x, s_hat * ones)
            x_next = x + (s_next - s_hat) * d
            if self.order == 2 and s_next > 0:
                d2, x0 = self._direction(denoise, guidance, x_next, s_next * ones)
                x_next = x + (s_next - s_hat) * (d + d2) / 2
            x = x_next

            if callback is not None:
                callback(i, float(s_next), x, x0)
        return x

    @staticmethod
    def _direction(denoise, guidance, x, sigma):
        if guidance is not None:
            return guidance(denoise, x, sigma)
        with torch.no_grad():
            x0 = denoise(x, sigma)
        return (x - x0) / _expand(sigma, x), x0


def _expand(sigma, x):
    return sigma.view(-1, *([1] * (x.ndim - 1)))
