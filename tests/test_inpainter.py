import pytest
import torch
from conftest import gaussian_denoiser

from simdps import Inpainter
from simdps.audio import crossfade, save_audio
from simdps.priors import DenoiserPrior

SR, HOP, N = 1000, 4, 100  # 0.4 s windows


def toy_prior():
    # A latent "autoencoder" that averages blocks of HOP samples into 3 channels
    encode = lambda a: a.mean(1, keepdim=True).unfold(-1, HOP, HOP).mean(-1).repeat(1, 3, 1)
    decode = lambda z: z.mean(1, keepdim=True).repeat_interleave(HOP, -1)
    return DenoiserPrior(gaussian_denoiser(0.0, 1.0), SR, N, encode, decode,
                         hop_length=HOP, pad_frames=8, sigma_max=10.0, cache_id="toy")


@pytest.fixture
def audio(generator):
    return torch.randn(1, 3 * SR, generator=generator)


GAP = (1.5, 1.55)


@pytest.mark.parametrize("method", ["multi", "solo", "dps", "search"])
def test_splices_into_input(method, audio):
    out = Inpainter(toy_prior(), buffer=4).inpaint(audio, SR, GAP, method=method, steps=10, seed=0)
    assert out.audio.shape == (1, *audio.shape)
    a, b = out.gap
    assert (a, b) == (1500, 1550)
    fade = round(0.1 * SR)
    torch.testing.assert_close(out.audio[0, :, : a - fade], audio[:, : a - fade])
    torch.testing.assert_close(out.audio[0, :, b + fade :], audio[:, b + fade :])
    assert not torch.allclose(out.audio[0, :, a:b], audio[:, a:b])


def test_raw_window(audio):
    out = Inpainter(toy_prior(), buffer=4).inpaint(audio, SR, GAP, raw=True, steps=5, n_samples=3)
    assert out.audio.shape == (3, 1, N * HOP)
    ws, we = out.window
    assert we - ws == N * HOP and ws % HOP == 0 and ws <= 1500 and we >= 1550
    assert out.latents.shape == (3, 3, N)


def test_seed(audio):
    inp = Inpainter(toy_prior(), buffer=4)
    a = inp.inpaint(audio, SR, GAP, steps=5, seed=3).audio
    b = inp.inpaint(audio, SR, GAP, steps=5, seed=3).audio
    assert torch.equal(a, b)


def test_search_fills_gap_with_candidate(audio):
    out = Inpainter(toy_prior(), buffer=4).inpaint(audio, SR, GAP, method="search", raw=True)
    s = out.candidates.signals[0]
    torch.testing.assert_close(out.latents[0], s)


def test_external_source(audio, generator):
    src = torch.randn(1, 2 * SR, generator=generator)
    out = Inpainter(toy_prior(), buffer=4).inpaint(audio, SR, GAP, source=src, steps=5)
    assert out.candidates.signals.shape[-1] == N


def test_window_at_edges(audio):
    inp = Inpainter(toy_prior(), buffer=4)
    assert inp.inpaint(audio, SR, (0.1, 0.15), steps=3).window[0] == 0
    assert inp.inpaint(audio, SR, (2.85, 2.9), steps=3).window[1] == 3 * SR


def test_short_input_is_padded(generator):
    audio = torch.randn(1, 300, generator=generator)
    out = Inpainter(toy_prior(), buffer=2).inpaint(audio, SR, (0.14, 0.16), steps=3, method="dps")
    assert out.audio.shape == (1, 1, 300)


def test_rejects_bad_requests(audio):
    inp = Inpainter(toy_prior())
    with pytest.raises(ValueError):
        inp.inpaint(audio, SR, (2.0, 4.0))
    with pytest.raises(ValueError):
        inp.inpaint(audio, SR, (1.0, 1.39))
    with pytest.raises(ValueError):
        inp.inpaint(audio, SR, GAP, method="magic")
    with pytest.raises(ValueError):
        inp.inpaint(audio, SR, GAP, unknown_option=1)


def test_resamples_input(generator):
    audio = torch.randn(2, 6000, generator=generator)
    out = Inpainter(toy_prior(), buffer=4).inpaint(audio, 2000, GAP, steps=3)
    assert out.sample_rate == SR and out.audio.shape == (1, 2, 3000)


def test_crossfade_is_continuous():
    a, b = torch.zeros(1, 100), torch.ones(1, 100)
    out = crossfade(a, b, 40, 60, 10)
    assert out[0, 29] == 0 and out[0, 70] == 0
    assert (out[0, 40:60] == 1).all()
    assert (out[0, 30:41].diff() >= 0).all() and (out[0, 59:71].diff() <= 0).all()


def test_seed_covers_stochastic_encoder(audio):
    prior = toy_prior()
    enc = prior._encode
    prior._encode = lambda a: enc(a) + 0.1 * torch.randn(1)
    inp = Inpainter(prior, buffer=4)
    a = inp.inpaint(audio, SR, GAP, steps=3, seed=5).audio
    state = torch.get_rng_state()
    b = inp.inpaint(audio, SR, GAP, steps=3, seed=5).audio
    assert torch.equal(a, b)
    assert torch.equal(state, torch.get_rng_state())


def test_track_encoding_is_cached(audio):
    prior = toy_prior()
    calls = []
    enc = prior._encode
    prior._encode = lambda a: calls.append(a.shape[-1]) or enc(a)
    inp = Inpainter(prior, buffer=4, cache_dir=None)
    inp.inpaint(audio, SR, GAP, steps=2)
    inp.inpaint(audio, SR, (1.0, 1.05), steps=2)
    assert calls.count(audio.shape[-1]) == 1
    inp.inpaint(audio * 0.5, SR, GAP, steps=2)
    assert calls.count(audio.shape[-1]) == 2


def test_seed_does_not_depend_on_cache(audio):
    prior = toy_prior()
    enc = prior._encode
    prior._encode = lambda a: enc(a) + 0.1 * torch.randn(1)
    warm = Inpainter(prior, buffer=4)
    warm.inpaint(audio, SR, GAP, steps=3, seed=5)
    a = warm.inpaint(audio, SR, GAP, steps=3, seed=5).audio
    b = Inpainter(prior, buffer=4).inpaint(audio, SR, GAP, steps=3, seed=5).audio
    assert torch.equal(a, b)


def test_disk_cache(audio, tmp_path):
    prior = toy_prior()
    calls = []
    enc = prior._encode
    prior._encode = lambda a: calls.append(a.shape[-1]) or enc(a)
    a = Inpainter(prior, buffer=4, cache_dir=tmp_path).inpaint(audio, SR, GAP, steps=2, seed=1).audio
    b = Inpainter(prior, buffer=4, cache_dir=tmp_path).inpaint(audio, SR, GAP, steps=2, seed=1).audio
    assert calls.count(audio.shape[-1]) == 1
    assert len(list(tmp_path.iterdir())) == 1
    assert torch.equal(a, b)


def test_speed_options(audio):
    inp = Inpainter(toy_prior(), buffer=4)
    out = inp.inpaint(audio, SR, GAP, raw=True, steps=4, window=0.2, pad=0, sigma_max=2.0, order=2, jacobian=False)
    assert out.audio.shape[-1] == round(0.2 * SR)
    assert out.latents.shape[-1] == round(0.2 * SR / HOP)
    assert torch.isfinite(out.audio).all()


@pytest.mark.parametrize("method", ["multi", "solo", "dps"])
def test_runs_requested_steps(method, audio):
    seen = []
    Inpainter(toy_prior(), buffer=4).inpaint(audio, SR, GAP, method=method, steps=17, seed=0,
                                             callback=lambda i, s, x, x0: seen.append(s))
    assert len(seen) == 17 and seen[-1] == 0


def test_runs_fixed_schedule(audio):
    seen = []
    Inpainter(toy_prior(), buffer=4).inpaint(audio, SR, GAP, sigmas=[5.0, 1.0, 0.3],
                                             callback=lambda i, s, x, x0: seen.append(s))
    assert seen == pytest.approx([1.0, 0.3, 0.0])


@pytest.mark.parametrize("config,steps", [(None, 100), ("warm_start_detailed", 40), ("warm_start_fast", 35)])
def test_config_steps(config, steps, audio):
    seen = []
    Inpainter(toy_prior(), buffer=4).inpaint(audio, SR, GAP, config=config, seed=0,
                                             callback=lambda i, s, x, x0: seen.append(s))
    assert len(seen) == steps


def test_schedule():
    inp = Inpainter(toy_prior(), buffer=4)
    assert inp.schedule() == 100
    assert inp.schedule("warm_start_fast") == 35
    inp.prior.sigmas = [5.0, 1.0, 0.3]
    assert inp.schedule("warm_start_fast") == 3


def test_unknown_config(audio):
    with pytest.raises(ValueError):
        Inpainter(toy_prior(), buffer=4).inpaint(audio, SR, GAP, config="slow")


def test_guide_by_time(audio):
    inp = Inpainter(toy_prior(), buffer=4)
    out = inp.inpaint(audio, SR, GAP, method="search", guide=0.8, raw=True)
    z = inp.encode(audio)
    fs = (1500 - out.window[0]) // HOP - 4
    # frame 4 of the regenerated span is the gap start, which takes the frame at 0.8 s
    torch.testing.assert_close(out.candidates.signals[0, :, fs + 4], z[:, 200])


def test_guide_from_source(audio, generator):
    src = torch.randn(1, 2 * SR, generator=generator)
    out = Inpainter(toy_prior(), buffer=4).inpaint(audio, SR, GAP, source=src, guide=[0.5, 1.2], steps=3)
    assert len(out.candidates) == 2


def test_pick_candidates(audio):
    inp = Inpainter(toy_prior(), buffer=4)
    ranked = inp.inpaint(audio, SR, GAP, method="search", n_samples=4, seed=0)
    out = inp.inpaint(audio, SR, GAP, candidates=[3, 1], steps=3, seed=0)
    assert out.candidates.starts.tolist() == ranked.candidates.starts[[3, 1]].tolist()


def test_search_returns_alternatives(audio):
    out = Inpainter(toy_prior(), buffer=4).inpaint(audio, SR, GAP, method="search", n_samples=3)
    assert out.audio.shape[0] == 3 and len(set(out.candidates.starts.tolist())) == 3


def test_dps_rejects_guide(audio):
    with pytest.raises(ValueError):
        Inpainter(toy_prior(), buffer=4).inpaint(audio, SR, GAP, method="dps", guide=0.8)


def test_folder_source(audio, generator, tmp_path):
    lib = tmp_path / "lib"
    (lib / "sub").mkdir(parents=True)
    for name in ["a.wav", "sub/b.flac"]:
        save_audio(lib / name, 0.5 * torch.rand(1, 2 * SR, generator=generator), SR)
    (lib / "notes.txt").write_text("not audio")
    inp = Inpainter(toy_prior(), buffer=4)
    out = inp.inpaint(audio, SR, GAP, source=lib, method="search", n_samples=6)
    assert set(out.candidates.sources) == {str(lib / "a.wav"), str(lib / "sub" / "b.flac")}
    assert (out.candidates.scores[1:] <= out.candidates.scores[:-1]).all()


def test_folder_source_cached(audio, generator, tmp_path, monkeypatch):
    lib = tmp_path / "lib"
    lib.mkdir()
    save_audio(lib / "a.wav", torch.randn(1, 2 * SR, generator=generator), SR)
    reads = []
    import simdps.inpainter as module
    load = module.load_audio
    monkeypatch.setattr(module, "load_audio", lambda *a: reads.append(a) or load(*a))
    for _ in range(2):
        Inpainter(toy_prior(), buffer=4, cache_dir=tmp_path / "c").inpaint(audio, SR, GAP, source=lib, method="search")
    assert len(reads) == 1


def test_empty_folder(audio, tmp_path):
    with pytest.raises(ValueError):
        Inpainter(toy_prior(), buffer=4).inpaint(audio, SR, GAP, source=tmp_path, method="search")


def test_listed_matches_are_distinct(audio):
    out = Inpainter(toy_prior(), buffer=4).inpaint(audio, SR, GAP, method="search", n_samples=4)
    s = out.candidates.starts.tolist()
    gap = round(0.05 * SR / HOP) + 8
    assert all(abs(a - b) >= gap for i, a in enumerate(s) for b in s[:i])


@pytest.mark.parametrize("match", ["latent", "audio", "bands"])
def test_search_lists_times(match, generator):
    # the audio features compare up to 5 s of context
    x = torch.randn(1, 15 * SR, generator=generator)
    found = Inpainter(toy_prior(), buffer=4).search(x, SR, (7.0, 7.05), n=3, match=match)
    assert len(found) == 3
    assert all(0 <= t <= 15 and 0 < c <= 1 and src is None for t, c, src in found)
    assert found[0][1] >= found[-1][1]


def test_search_times_are_guides(audio):
    # a time from search(), passed back as a guide, gives the same candidate
    inp = Inpainter(toy_prior(), buffer=4)
    t, _, _ = inp.search(audio, SR, GAP, n=1)[0]
    auto = inp.inpaint(audio, SR, GAP, method="search", raw=True)
    guided = inp.inpaint(audio, SR, GAP, method="search", raw=True, guide=t)
    torch.testing.assert_close(guided.latents, auto.latents)


def test_search_other_recording(audio, generator):
    other = torch.randn(1, 2 * SR, generator=generator)
    found = Inpainter(toy_prior(), buffer=4).search(audio, SR, GAP, source=other, n=2)
    assert len(found) == 2 and found[0][2] is other


def test_guides_from_several_recordings(audio, generator):
    other = torch.randn(1, 2 * SR, generator=generator)
    out = Inpainter(toy_prior(), buffer=4).inpaint(audio, SR, GAP, steps=3, seed=0,
                                                 guide=[(0.5, other), (1.2, None)])
    assert len(out.candidates) == 2 and out.candidates.sources == ["source", None]


def test_guide_weights(audio, monkeypatch):
    import simdps.inpainter as m

    seen = {}
    real = m.Candidates

    def spy(*a, **k):
        seen["w"] = k.get("weights")
        return real(*a, **k)

    monkeypatch.setattr(m, "Candidates", spy)
    out = Inpainter(toy_prior(), buffer=4).inpaint(audio, SR, GAP, steps=2, guide=[0.3, 2.0], weights=[0.2, 0.8])
    torch.testing.assert_close(seen["w"], torch.tensor([0.8, 0.2]))
    assert out.candidates.starts[0] > out.candidates.starts[1]


def test_shares():
    from simdps.inpainter import _shares

    scores = torch.tensor([0.5, 0.5, 0.9])
    assert _shares(scores, [None, None, None]) is None
    w = _shares(scores, [0.5, None, None])
    assert w[0] == pytest.approx(0.5) and w.sum() == pytest.approx(1.0) and w[2] > w[1]
    torch.testing.assert_close(_shares(scores, [2.0, 2.0, 4.0]), torch.tensor([0.25, 0.25, 0.5]))


def test_weights_need_guides(audio):
    with pytest.raises(ValueError):
        Inpainter(toy_prior(), buffer=4).inpaint(audio, SR, GAP, weights=[1.0])


def test_checkpoints(audio):
    inp = Inpainter(toy_prior(), buffer=4)
    out = inp.inpaint(audio, SR, GAP, steps=12, n_samples=2, seed=0, checkpoints=3)
    assert [c.step for c in out.checkpoints] == [3, 6, 9]
    assert all(c.latents.shape == (2, 3, N + 8) for c in out.checkpoints)
    assert out.checkpoints[0].sigma > out.checkpoints[-1].sigma > 0
    y = inp.decode(out.checkpoints[-1].latents, audio, SR, out)
    assert y.shape == out.audio.shape
    a, b = out.gap
    fade = round(0.1 * SR)
    torch.testing.assert_close(y[0, :, : a - fade], audio[:, : a - fade])


def test_restart(audio):
    inp = Inpainter(toy_prior(), buffer=4)
    first = inp.inpaint(audio, SR, GAP, steps=12, seed=0, checkpoints=3)
    c = first.checkpoints[1]
    seen = []
    a = inp.inpaint(audio, SR, GAP, steps=12, seed=1, restart=c, n_samples=2, callback=lambda i, *_: seen.append(i))
    b = inp.inpaint(audio, SR, GAP, steps=12, seed=1, restart=c, n_samples=2)
    assert seen == list(range(c.step, 12))
    torch.testing.assert_close(a.audio, b.audio)
    assert not torch.allclose(a.audio[0], a.audio[1])
    with pytest.raises(ValueError):
        inp.inpaint(audio, SR, GAP, steps=5, restart=c)


def test_trace(audio):
    seen = []
    Inpainter(toy_prior(), buffer=4).inpaint(audio, SR, GAP, steps=4, n_samples=2, guide=[0.3, 2.0],
                                             trace=lambda i, s, x0, g: seen.append((x0.shape, g.shape)))
    assert len(seen) == 4
    xs, gs = seen[0]
    assert xs[0] == 2 and gs[0] == 2 and xs[1:] == gs[1:]


def test_disk_cache_needs_an_id(audio, tmp_path):
    prior = toy_prior()
    prior.cache_id = None
    Inpainter(prior, buffer=4, cache_dir=tmp_path).encode(audio, seed=0)
    assert not list(tmp_path.iterdir())


def test_disk_cache_keeps_autoencoders_apart(audio, tmp_path):
    a, b = toy_prior(), toy_prior()
    b.cache_id = "other"
    b._encode = lambda x: 2 * a._encode(x)
    za = Inpainter(a, buffer=4, cache_dir=tmp_path).encode(audio, seed=0)
    zb = Inpainter(b, buffer=4, cache_dir=tmp_path).encode(audio, seed=0)
    torch.testing.assert_close(zb, 2 * za)
