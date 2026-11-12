"""Tests with the real ACE-Step weights (about 5 GB). Run with ``pytest -m slow``."""

import pytest
import torch

import simdps

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def model():
    return simdps.load("acestep")


def test_round_trip(model):
    prior = model.prior
    t = torch.arange(3 * prior.sample_rate) / prior.sample_rate
    audio = 0.3 * torch.sin(2 * torch.pi * 220 * t)[None, None]
    z = prior.encode(audio)
    assert z.shape == (1, 64, 3 * prior.sample_rate // prior.hop_length)
    y = prior.decode(z)
    assert y.shape[:2] == (1, 2)
    n = min(y.shape[-1], audio.shape[-1])
    err = (y[0, 0, :n].cpu() - audio[0, 0, :n]).pow(2).mean() / audio.pow(2).mean()
    assert err < 0.1


def test_denoiser_shrinks_noise(model):
    prior = model.prior
    z = torch.zeros(1, 64, prior.length + prior.pad_frames, device=model.device)
    noise = torch.randn_like(z)
    with torch.no_grad():
        x0 = prior.denoise(z + 10 * noise, torch.tensor([10.0], device=model.device))
    assert x0.shape == z.shape
    assert x0.std() < (10 * noise).std()


def test_inpaint(model):
    sr = 48000
    t = torch.arange(20 * sr) / sr
    audio = 0.2 * torch.sin(2 * torch.pi * 330 * t)[None] * (1 + torch.sin(2 * torch.pi * t))
    out = model.inpaint(audio, sr, (9.0, 11.0), steps=4, seed=0)
    assert out.audio.shape == (1, 1, 20 * sr)
    assert torch.isfinite(out.audio).all()
    assert len(out.candidates) == 15
