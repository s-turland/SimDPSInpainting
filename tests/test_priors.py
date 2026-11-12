import torch
from conftest import gaussian_denoiser

from simdps.priors import DenoiserPrior, EDMPrior, FlowPrior

STD = 0.4


def _sigma():
    return torch.tensor([0.01, 0.3, 2.0, 50.0])


def test_edm_prior_with_zero_network_is_gaussian_denoiser(generator):
    # c_skip alone is the exact denoiser of N(0, sigma_data^2)
    prior = EDMPrior(lambda x, c: torch.zeros_like(x), sigma_data=STD, sample_rate=1, length=8)
    x = torch.randn(4, 1, 8, generator=generator)
    torch.testing.assert_close(prior.denoise(x, _sigma()), gaussian_denoiser(0.0, STD)(x, _sigma()))


def test_flow_prior_with_exact_velocity(generator):
    def velocity(xt, t):
        t = t.view(-1, 1, 1)
        x0 = (1 - t) * STD**2 / ((1 - t) ** 2 * STD**2 + t**2) * xt
        return (xt - x0) / t

    prior = FlowPrior(velocity, sample_rate=1, length=8)
    x = torch.randn(4, 1, 8, generator=generator)
    torch.testing.assert_close(prior.denoise(x, _sigma()), gaussian_denoiser(0.0, STD)(x, _sigma()))


def test_denoiser_prior_passes_through():
    prior = DenoiserPrior(lambda x, s: x * 0, 16000, 8, encode=lambda a: a * 2, decode=lambda z: z / 2,
                          hop_length=4)
    a = torch.ones(1, 1, 8)
    assert torch.equal(prior.decode(prior.encode(a)), a)
    assert prior.hop_length == 4
    assert torch.equal(DenoiserPrior(None, 1, 1).encode(a), a)
