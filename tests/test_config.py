import os

import pytest
import torch
from conftest import gaussian_denoiser
from test_inpainter import GAP, SR, toy_prior

import simdps
from simdps import Inpainter, load_config
from simdps.config import available, merge, with_options
from simdps.guidance import Candidates, Guidance, Observation, compressed_stft, gap_mask
from simdps.priors import DenoiserPrior, build_prior

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def audio(generator):
    return torch.randn(1, 3 * SR, generator=generator)


def make_toy(device=None, dtype=None):
    return toy_prior()


# What the paper reports. A change here changes what the code reproduces.
LATENT = dict(
    observation={"zeta": 0.5},
    guidance={"project": True, "jacobian": True},
)
SAMPLER = {"steps": 100, "sigma_min": 1e-3, "sigma_max": 20, "rho": 7, "churn": 10, "order": 1}
PIANO_SAMPLER = {"steps": 50, "sigma_min": 1e-5, "sigma_max": 8, "rho": 10, "churn": 10, "churn_range": [0, 10],
                 "order": 2, "start": "noise"}


def test_multi_settings():
    c = load_config("ismir26_multi")
    assert c["method"] == "multi" and c["buffer"] == 20
    assert c["search"] == {"n": 15, "context": 5, "crossfade": 5}
    assert c["candidates"] == {"weight": 0.1, "combine": "weighted", "distance": "lowpass", "cutoff": 8,
                               "taper": 0.5, "stop": 0.5, "fade": 0.5}
    assert c["sampler"] == {**SAMPLER, "start": "candidate"}
    assert all(c[k] == v for k, v in LATENT.items())


def test_solo_settings():
    c = load_config("ismir26_solo")
    assert c["method"] == "solo" and c["buffer"] == 20
    assert c["candidates"] == {"weight": 0.3, "combine": "solo", "distance": "l2", "cutoff": 15, "taper": None,
                               "stop": 1e-3, "fade": 0.1}
    assert c["sampler"] == {**SAMPLER, "start": "noise"}
    assert all(c[k] == v for k, v in LATENT.items())


def test_dps_settings():
    c = load_config("ismir26_dps")
    assert c["method"] == "dps" and "candidates" not in c
    assert c["sampler"] == {**SAMPLER, "start": "noise"}


@pytest.mark.parametrize("name,weight", [("ismir26_piano_strict", 0.15), ("ismir26_piano_relaxed", 0.04)])
def test_piano_settings(name, weight):
    c = load_config(name)
    assert c["prior"] == {"name": "cqtdiff", "checkpoint": None} and c["buffer"] == 0
    assert c["observation"] == {"zeta": 0.3, "loss": "comp_stft", "compression": 0.3}
    assert c["candidates"]["weight"] == weight and c["candidates"]["combine"] == "min"
    assert c["guidance"] == {"project": False, "jacobian": True, "joint": True}
    assert c["sampler"] == PIANO_SAMPLER
    assert c["search"]["rate"] == 16000 and c["search"]["n"] == 5
    assert c["search"]["stft"]["context"] == 2.0 and c["search"]["chroma"]["context"] == 5.0
    assert c["search"]["chroma"]["weight"] == 1.5


def test_warm_starts_list_only_their_changes():
    assert load_config("warm_start_detailed") == {"sampler": {"steps": 40, "sigma_max": 1, "start": "candidate"}}
    assert load_config("warm_start_fast") == {"sampler": {"steps": 35, "sigma_max": 1, "start": "candidate"},
                                              "guidance": {"jacobian": False}}


def test_every_config_is_listed():
    names = available()
    assert {"ismir26_multi", "ismir26_solo", "ismir26_dps", "ismir26_search", "solo_lowpass", "ismir26_piano_strict",
            "ismir26_piano_relaxed", "ismir26_piano_dps", "ismir26_piano_search"} == set(names["experiment"])
    assert {"warm_start_detailed", "warm_start_fast"} == set(names["speed"])


def test_merging_and_assignment():
    c = load_config("ismir26_multi", "warm_start_fast", "candidates.weight=0.2", {"sampler": {"rho": 9}})
    assert c["sampler"]["steps"] == 35 and c["sampler"]["sigma_min"] == 1e-3 and c["sampler"]["rho"] == 9
    assert c["candidates"]["weight"] == 0.2 and c["candidates"]["cutoff"] == 8
    assert c["guidance"]["jacobian"] is False
    assert load_config("a.b=1e-3") == {"a": {"b": 1e-3}}
    assert merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 3}}) == {"a": {"b": 3, "c": 2}}


def test_yaml_file(tmp_path):
    p = tmp_path / "mine.yaml"
    p.write_text("sampler:\n  sigma_min: 1e-4\n  steps: 7\n")
    assert load_config(p) == {"sampler": {"sigma_min": 1e-4, "steps": 7}}
    assert isinstance(load_config(str(p))["sampler"]["sigma_min"], float)


def test_unknown_config_lists_the_names():
    with pytest.raises(ValueError, match="ismir26_multi"):
        load_config("nothing")


def test_options_go_to_their_sections():
    c = with_options({"sampler": {"steps": 3}}, {"weight": 0.2, "steps": 5, "n": 2, "jacobian": False, "pad": 0})
    assert c["candidates"]["weight"] == 0.2 and c["sampler"]["steps"] == 5 and c["search"]["n"] == 2
    assert c["guidance"]["jacobian"] is False and c["pad"] == 0
    with pytest.raises(ValueError):
        with_options({}, {"nothing": 1})


@pytest.mark.parametrize("name", ["ismir26_multi", "ismir26_solo", "solo_lowpass", "ismir26_dps", "ismir26_search"])
def test_latent_experiments_run(name, audio):
    inp = Inpainter(toy_prior(), config=[name, "buffer=4"])
    out = inp.inpaint(audio, SR, GAP, steps=3, seed=0)
    assert torch.isfinite(out.audio).all()
    assert (out.candidates is None) == (name == "ismir26_dps")


def _terms(monkeypatch):
    import simdps.inpainter as m

    seen = {}
    real = m.Candidates

    def spy(*a, **k):
        seen.update(k)
        return real(*a, **k)

    monkeypatch.setattr(m, "Candidates", spy)
    return seen


def test_solo_switches_to_lowpass(audio, monkeypatch):
    seen = _terms(monkeypatch)
    inp = Inpainter(toy_prior(), config=["ismir26_solo", "buffer=4"])
    inp.inpaint(audio, SR, GAP, steps=2, config="candidates.distance=lowpass")
    assert seen["distance"] == "lowpass" and seen["cutoff"] == 15
    inp.inpaint(audio, SR, GAP, steps=2, distance="l2")
    assert seen["distance"] == "l2"
    inp.inpaint(audio, SR, GAP, steps=2, config="solo_lowpass")
    assert seen["distance"] == "lowpass" and seen["weight"] == 0.35 and seen["stop"] == 0.5


def test_method_keeps_the_schedule_of_the_config(audio):
    inp = Inpainter(toy_prior(), config=[{"sampler": {"steps": 2}}, "buffer=4"])
    seen = []
    inp.inpaint(audio, SR, GAP, method="solo", callback=lambda i, s, x, x0: seen.append(s))
    assert len(seen) == 2
    assert inp.settings("solo")["candidates"]["distance"] == "l2"
    assert inp.settings("multi")["candidates"]["distance"] == "lowpass"


def test_solo_starts_from_noise_and_warm_start_from_the_match():
    inp = Inpainter(toy_prior(), buffer=4)
    assert inp.settings("solo")["sampler"]["start"] == "noise"
    assert inp.settings("multi")["sampler"]["start"] == "candidate"
    assert inp.settings("solo", "warm_start_fast")["sampler"]["start"] == "candidate"


def test_warm_start_needs_candidates(audio):
    with pytest.raises(ValueError, match="warm start"):
        Inpainter(toy_prior(), buffer=4).inpaint(audio, SR, GAP, method="dps", config="warm_start_fast")


def test_prior_noise_range_is_kept_without_a_config(audio):
    # the toy prior starts at 10, not at the 20 of ismir26_multi
    seen = []
    inp = Inpainter(toy_prior(), buffer=4)
    inp.inpaint(audio, SR, GAP, steps=3, callback=lambda i, s, x, x0: seen.append(s))
    assert seen[0] < 10
    assert "sigma_max" not in inp.config["sampler"]


def test_waveform_prior_defaults_to_no_buffer():
    prior = DenoiserPrior(gaussian_denoiser(0.0, 1.0), SR, 1000)
    assert Inpainter(prior).buffer == 0


def test_plug_in_own_model(monkeypatch):
    monkeypatch.syspath_prepend(os.path.dirname(os.path.abspath(__file__)))
    prior = build_prior({"target": "test_config:make_toy"})
    assert prior.hop_length == 4
    with pytest.raises(ValueError):
        build_prior({"name": "nothing"})


def test_load_from_config(monkeypatch):
    monkeypatch.syspath_prepend(os.path.dirname(os.path.abspath(__file__)))
    model = simdps.load({"prior": {"target": "test_config:make_toy"}, "buffer": 4}, device="cpu")
    assert isinstance(model, Inpainter) and model.buffer == 4 and model.config["method"] == "multi"
    with pytest.raises(ValueError, match="which model"):
        simdps.load("warm_start_fast", device="cpu")


# --- the piano's guidance ---


def test_compressed_stft_shape_and_gradient(generator):
    x = torch.randn(2, 1, 4096, generator=generator)
    x[..., 1000:3000] = 0
    x.requires_grad_(True)
    X = compressed_stft(x, 0.3)
    assert X.shape[:2] == (2, 1) and X.is_complex()
    X.abs().pow(2).mean().backward()
    assert torch.isfinite(x.grad).all()


def _waveform_problem(generator):
    T, gap = 4096, (1500, 2500)
    z = torch.randn(1, 1, T, generator=generator)
    mask = gap_mask(T, *gap)
    cands = torch.randn(2, 1, T, generator=generator)
    return z, mask, cands, gap


def test_comp_stft_terms_are_zero_at_the_target(generator):
    z, mask, cands, gap = _waveform_problem(generator)
    assert Observation(mask * z, mask, loss="comp_stft").loss(z) == pytest.approx(0.0, abs=1e-8)
    term = Candidates(cands, torch.ones(2), gap, distance="comp_stft", combine="min", taper=None, stop=0, fade=0)
    assert term.loss(cands[:1]) == pytest.approx(0.0, abs=1e-8)
    assert term.loss(z) > 0


def test_joint_gradient(generator):
    z, mask, cands, gap = _waveform_problem(generator)
    obs = Observation(mask * z, mask, zeta=0.3, loss="comp_stft")
    term = Candidates(cands, torch.ones(2), gap, weight=0.15, distance="comp_stft", combine="min", taper=None,
                      stop=0, fade=0)
    denoise = lambda x, s: 0.9 * x
    x = torch.randn(1, 1, 4096, generator=generator)
    sigma = torch.full((1,), 2.0)
    g, _ = Guidance([obs, term], project=False, joint=True)(denoise, x, sigma)
    d = (x - denoise(x, sigma)) / 2.0
    rms = (g - d).pow(2).mean().sqrt()
    assert rms == pytest.approx(0.3, rel=1e-3)  # the whole gradient is scaled by zeta

    # the candidates weigh in relative to the observation
    obs_only, _ = Guidance([obs], project=False, joint=True)(denoise, x, sigma)
    assert not torch.allclose(g, obs_only)


def test_piano_schedule_and_search_with_a_toy_denoiser(generator):
    x = torch.randn(1, 20 * 16000, generator=generator) * 0.1
    prior = DenoiserPrior(gaussian_denoiser(0.0, 0.1), 16000, 8192, sigma_min=1e-5, sigma_max=8.0)
    cfg = load_config("ismir26_piano_strict")
    inp = Inpainter(prior, config=cfg)
    seen = []
    out = inp.inpaint(x, 16000, (10.0, 10.2), steps=2, seed=0, callback=lambda i, s, a, b: seen.append(s))
    assert len(seen) == 2 and len(out.candidates) == 5
    assert torch.isfinite(out.audio).all()


# --- the piano model ---


def test_piano_needs_its_checkpoint():
    with pytest.raises(ValueError, match="prior.checkpoint"):
        build_prior({"name": "cqtdiff", "checkpoint": None})


@pytest.mark.parametrize("key", [None, "ema", "network"])
def test_piano_checkpoint_formats(key, tmp_path):
    pytest.importorskip("einops")
    from simdps.priors.cqtdiff import NETWORK, CQTDiff, build_network

    net = build_network(NETWORK)
    state = {("module." + k): v for k, v in net.state_dict().items()}
    path = tmp_path / "piano.pt"
    torch.save(state if key is None else {key: state, "other": 1}, path)
    prior = CQTDiff(checkpoint=str(path))
    assert prior.hop_length == 1 and prior.sample_rate == 44100 and prior.length == 262144
    a = next(iter(prior.net.state_dict().values()))
    b = next(iter(net.state_dict().values()))
    assert torch.equal(a, b)
    assert prior.encode(torch.zeros(1, 2, 16)).shape == (1, 1, 16)


def test_piano_rejects_the_wrong_network(tmp_path):
    pytest.importorskip("einops")
    from simdps.priors.cqtdiff import CQTDiff

    path = tmp_path / "other.pt"
    torch.save({"something": torch.zeros(1)}, path)
    with pytest.raises(ValueError, match="does not fit"):
        CQTDiff(checkpoint=str(path))
