import pytest
import torch
from conftest import gaussian_denoiser

from simdps import EDMSampler
from simdps.guidance import Guidance, Observation, gap_mask


def test_schedule():
    s = EDMSampler(steps=50, sigma_min=1e-3, sigma_max=20).sigmas()
    assert len(s) == 51
    assert s[0] == pytest.approx(20, rel=1e-5)
    assert s[-2] == pytest.approx(1e-3, rel=1e-4)
    assert s[-1] == 0
    assert (s[1:] < s[:-1]).all()


def test_churn_range():
    s = EDMSampler(steps=100, churn=10, churn_range=(0.0, 5.0))
    assert s._gamma(torch.tensor(8.0)) == 0
    assert s._gamma(torch.tensor(2.0)) == pytest.approx(0.1)
    assert EDMSampler(steps=2, churn=10)._gamma(torch.tensor(1.0)) == pytest.approx(2**0.5 - 1)


@pytest.mark.parametrize("order, churn", [(1, 0.0), (2, 0.0), (2, 10.0)])
def test_samples_gaussian_prior(order, churn, generator):
    mean, std = 0.3, 0.5
    sampler = EDMSampler(steps=60, sigma_max=20, sigma_min=1e-3, order=order, churn=churn)
    x = 20 * torch.randn(4000, 1, 8, generator=generator)
    out = sampler.sample(gaussian_denoiser(mean, std), x, generator=generator)
    assert out.mean().item() == pytest.approx(mean, abs=0.02)
    assert out.std().item() == pytest.approx(std, rel=0.05 if order == 2 else 0.1)


def test_deterministic_without_churn(generator):
    sampler = EDMSampler(steps=20, churn=0.0)
    x = 20 * torch.randn(2, 3, 16, generator=generator)
    a = sampler.sample(gaussian_denoiser(0.0, 1.0), x)
    b = sampler.sample(gaussian_denoiser(0.0, 1.0), x)
    assert torch.equal(a, b)


def test_generator_makes_churn_reproducible():
    sampler = EDMSampler(steps=20, churn=10)
    x = torch.ones(1, 1, 32)
    a = sampler.sample(gaussian_denoiser(0.0, 1.0), x, generator=torch.Generator().manual_seed(1))
    b = sampler.sample(gaussian_denoiser(0.0, 1.0), x, generator=torch.Generator().manual_seed(1))
    assert torch.equal(a, b)


@pytest.mark.parametrize("project", [False, True])
def test_guidance_matches_observation(project, generator):
    T, gap = 64, (24, 40)
    y = torch.randn(1, 1, T, generator=generator)
    mask = gap_mask(T, *gap)
    guide = Guidance([Observation(mask * y, mask, zeta=0.5)], project=project)
    sampler = EDMSampler(steps=100, churn=0.0)
    x = 20 * torch.randn(64, 1, T, generator=generator)
    out = sampler.sample(gaussian_denoiser(0.0, 1.0), x, guide)

    err = ((out - y) * mask).pow(2).sum() / mask.sum() / len(out)
    free = sampler.sample(gaussian_denoiser(0.0, 1.0), x)
    err_free = ((free - y) * mask).pow(2).sum() / mask.sum() / len(out)
    assert err < 0.05 * err_free
    # frames are independent under this prior, so without projection the gap is
    # left to it; projecting against the prior direction also shrinks the gap
    gap_std = out[..., gap[0] : gap[1]].std().item()
    if project:
        assert gap_std < 0.8
    else:
        assert gap_std == pytest.approx(1.0, rel=0.15)


def test_callback_sees_every_step():
    seen = []
    EDMSampler(steps=7).sample(
        gaussian_denoiser(0.0, 1.0), torch.ones(1, 1, 4), callback=lambda i, s, x, x0: seen.append((i, s))
    )
    assert [i for i, _ in seen] == list(range(7))
    assert seen[-1][1] == 0


def test_fixed_sigmas():
    s = EDMSampler(sigmas=[5.0, 1.0, 0.2])
    torch.testing.assert_close(s.sigmas(), torch.tensor([5.0, 1.0, 0.2, 0.0]))
    assert s.steps == 3 and s.sigma_max == 5.0
    seen = []
    s.sample(gaussian_denoiser(0.0, 1.0), torch.ones(1, 1, 4), callback=lambda i, sg, x, x0: seen.append(sg))
    assert seen == pytest.approx([1.0, 0.2, 0.0])


def test_euler_matches_flow_euler_on_same_times(generator):
    # an Euler step in sigma is the rectified-flow Euler step at t = sigma / (1 + sigma)
    std = 0.5
    x0_of = lambda xt, t: (1 - t) * std**2 / ((1 - t) ** 2 * std**2 + t**2) * xt
    times = [0.95, 0.8, 0.5, 0.3]
    xt = torch.randn(3, 1, 8, generator=generator)
    x = xt / (1 - times[0])
    for t, t2 in zip(times, times[1:] + [0.0]):
        xt = xt + (t2 - t) * (xt - x0_of(xt, t)) / t
    sigmas = [t / (1 - t) for t in times]
    out = EDMSampler(sigmas=sigmas, churn=0.0).sample(gaussian_denoiser(0.0, std), x)
    torch.testing.assert_close(out, xt, rtol=1e-4, atol=1e-5)


def test_start_resumes_the_schedule():
    seen = []
    s = EDMSampler(steps=7, churn=0)
    s.sample(gaussian_denoiser(0.0, 1.0), torch.ones(1, 1, 4), callback=lambda i, sg, x, x0: seen.append(i), start=4)
    assert seen == [4, 5, 6]


def test_schedule_needs_two_steps():
    with pytest.raises(ValueError, match="at least 2"):
        EDMSampler(steps=1)
    assert EDMSampler(sigmas=[1.0]).sigmas().tolist() == [1.0, 0.0]
