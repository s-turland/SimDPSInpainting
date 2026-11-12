import math

import pytest
import torch
from conftest import gaussian_denoiser

from simdps.guidance import Candidates, Guidance, Observation, _deconflict, gap_mask, tukey

T, GAP = 96, (32, 64)


def _rms(g):
    return (g.pow(2).mean()).sqrt().item()


def test_gradient_rms_equals_weight(generator):
    y = torch.randn(1, 4, T, generator=generator)
    mask = gap_mask(T, *GAP)
    x = 3 * torch.randn(1, 4, T, generator=generator)
    sigma = torch.tensor([3.0])
    d0, _ = Guidance([])(gaussian_denoiser(0.0, 1.0), x, sigma)
    for zeta in (0.1, 0.5, 2.0):
        d, _ = Guidance([Observation(mask * y, mask, zeta)], project=False)(gaussian_denoiser(0.0, 1.0), x, sigma)
        assert _rms(d - d0) == pytest.approx(zeta, rel=1e-4)


def test_batch_items_are_independent(generator):
    y = torch.randn(1, 2, T, generator=generator)
    mask = gap_mask(T, *GAP)
    x = torch.randn(3, 2, T, generator=generator)
    sigma = torch.full((3,), 2.0)
    guide = Guidance([Observation(mask * y, mask)])
    d, _ = guide(gaussian_denoiser(0.0, 1.0), x, sigma)
    d1, _ = guide(gaussian_denoiser(0.0, 1.0), x[1:2], sigma[:1])
    torch.testing.assert_close(d[1:2], d1)


def test_length_crops_padding(generator):
    y = torch.randn(1, 1, T, generator=generator)
    mask = gap_mask(T, *GAP)
    x = torch.randn(1, 1, T + 10, generator=generator)
    sigma = torch.tensor([1.0])
    d, _ = Guidance([Observation(mask * y, mask)], project=False, length=T)(lambda x, s: x / 2, x, sigma)
    d0 = x / 2
    torch.testing.assert_close(d[..., T:], d0[..., T:])


def test_deconflict():
    ref = torch.tensor([[1.0, 0.0]])
    torch.testing.assert_close(_deconflict(torch.tensor([[-1.0, 1.0]]), ref), torch.tensor([[0.0, 1.0]]))
    torch.testing.assert_close(_deconflict(torch.tensor([[1.0, 1.0]]), ref), torch.tensor([[1.0, 1.0]]))


def test_projection_removes_conflict(generator):
    y = torch.randn(1, 1, T, generator=generator)
    mask = gap_mask(T, *GAP)
    x = torch.randn(1, 1, T, generator=generator)
    sigma = torch.tensor([0.5])
    denoise = gaussian_denoiser(0.0, 1.0)
    d0, _ = Guidance([])(denoise, x, sigma)
    d, _ = Guidance([Observation(mask * y, mask, zeta=5.0)])(denoise, x, sigma)
    assert torch.dot((d - d0).flatten(), d0.flatten()) >= -1e-5


def _candidates(generator, **kw):
    c = torch.randn(3, 4, T, generator=generator)
    return Candidates(c, torch.tensor([0.9, 0.5, 0.1]), GAP, **kw)


def test_weight_schedule(generator):
    cand = _candidates(generator, weight=0.2, stop=0.5, fade=0.5, sigma_range=(0.0, 20.0))
    assert cand.weight(torch.tensor([0.5])) is None
    assert cand.weight(torch.tensor([20.0])).item() == pytest.approx(0.2, rel=1e-4)
    assert cand.weight(torch.tensor([5.0])).item() == pytest.approx(0.2 * math.sqrt(5 / 20), rel=1e-4)
    flat = _candidates(generator, weight=0.2, fade=0.0)
    assert flat.weight(torch.tensor([1.0])).item() == pytest.approx(0.2)


@pytest.mark.parametrize("distance", ["lowpass", "magnitude", "l2"])
def test_distance_is_zero_on_candidate(distance, generator):
    cand = _candidates(generator, distance=distance)
    x0 = cand.candidates[1:2].clone()
    x0[..., : GAP[0]] = 7.0
    assert cand.dist(x0, cand.candidates[1]).item() == pytest.approx(0.0, abs=1e-6)
    assert cand.dist(x0, cand.candidates[0]).item() > 0.1


@pytest.mark.parametrize("distance", ["lowpass", "magnitude"])
def test_lowpass_ignores_detail_above_cutoff(distance, generator):
    cand = _candidates(generator, distance=distance, cutoff=4, taper=None)
    n = GAP[1] - GAP[0]
    t = torch.arange(n)
    x0 = cand.candidates[:1].clone()
    x0[..., GAP[0] : GAP[1]] += torch.cos(2 * math.pi * 9 * t / n)
    assert cand.dist(x0, cand.candidates[0]).item() == pytest.approx(0.0, abs=1e-5)
    x0[..., GAP[0] : GAP[1]] += torch.cos(2 * math.pi * 2 * t / n)
    assert cand.dist(x0, cand.candidates[0]).item() > 0.01


def test_magnitude_ignores_circular_shift(generator):
    kw = dict(cutoff=8, taper=None)
    mag = _candidates(generator, distance="magnitude", **kw)
    low = Candidates(mag.candidates, mag.scores, GAP, distance="lowpass", **kw)
    x0 = mag.candidates[:1].clone()
    x0[..., GAP[0] : GAP[1]] = x0[..., GAP[0] : GAP[1]].roll(5, -1)
    assert mag.dist(x0, mag.candidates[0]).item() == pytest.approx(0.0, abs=1e-5)
    assert low.dist(x0, low.candidates[0]).item() > 0.01


def test_combine_modes(generator):
    x0 = torch.randn(1, 4, T, generator=generator)
    base = _candidates(generator)
    d = torch.stack([base.dist(x0, c) for c in base.candidates])[:, 0]
    w = torch.softmax(base.scores, 0)
    expected = {"weighted": (w * d).sum(), "solo": d[0], "min": d.min(), "mean": d.mean()}
    for mode, value in expected.items():
        c = Candidates(base.candidates, base.scores, GAP, combine=mode)
        assert c.loss(x0).item() == pytest.approx(value.item(), rel=1e-5)


def test_distance_gradients():
    g = torch.Generator().manual_seed(1)
    c = torch.randn(2, 2, 40, generator=g, dtype=torch.double)
    for distance in ("lowpass", "magnitude", "l2"):
        cand = Candidates(c, torch.tensor([1.0, 0.5]), (10, 30), distance=distance, cutoff=5)
        cand.mask = cand.mask.double()
        if cand.window is not None:
            cand.window = cand.window.double()
        x0 = torch.randn(1, 2, 40, generator=g, dtype=torch.double, requires_grad=True)
        assert torch.autograd.gradcheck(cand.loss, (x0,))


def test_tukey():
    w = tukey(40, 0.5)
    assert w[0] == 0 and w[-1] == 0
    assert (w[10:30] == 1).all()
    torch.testing.assert_close(w, w.flip(0))


def test_rejects_unknown_options(generator):
    with pytest.raises(ValueError):
        _candidates(generator, combine="best")
    with pytest.raises(ValueError):
        _candidates(generator, distance="cosine")


def test_batched_backward_matches_separate(generator):
    net = torch.nn.Conv1d(4, 4, 9, padding=4)
    denoise = lambda x, s: torch.tanh(net(x)) + x / (1 + s.view(-1, 1, 1))
    y = torch.randn(1, 4, T, generator=generator)
    mask = gap_mask(T, *GAP)
    x = torch.randn(2, 4, T, generator=generator)
    sigma = torch.full((2,), 3.0)
    terms = [Observation(mask * y, mask), _candidates(generator)]
    d1, _ = Guidance(terms, batched=True)(denoise, x, sigma)
    d2, _ = Guidance(terms, batched=False)(denoise, x, sigma)
    torch.testing.assert_close(d1, d2)


def test_jacobian_free_matches_for_identity_denoiser(generator):
    # with x0 = x the two gradients coincide
    y = torch.randn(1, 4, T, generator=generator)
    mask = gap_mask(T, *GAP)
    x = torch.randn(1, 4, T, generator=generator)
    sigma = torch.tensor([2.0])
    terms = [Observation(mask * y, mask), _candidates(generator)]
    d1, _ = Guidance(terms, jacobian=True)(lambda x, s: x * 1.0, x, sigma)
    d2, _ = Guidance(terms, jacobian=False)(lambda x, s: x * 1.0, x, sigma)
    torch.testing.assert_close(d1, d2)


def test_jacobian_free_does_not_backpropagate(generator):
    net = torch.nn.Conv1d(4, 4, 3, padding=1)
    y = torch.randn(1, 4, T, generator=generator)
    mask = gap_mask(T, *GAP)
    x = torch.randn(1, 4, T, generator=generator)
    Guidance([Observation(mask * y, mask)], jacobian=False)(lambda x, s: net(x), x, torch.tensor([1.0]))
    assert net.weight.grad is None
