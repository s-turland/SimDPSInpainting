import torch
from conftest import gaussian_denoiser

from simdps import Inpainter
from simdps.priors import DenoiserPrior
from simdps.search import BandFeature, ChromaFeature, STFTFeature, band_filterbank, chroma_filterbank, waveform_search

SR = 8000


def _music(generator, seconds=20):
    notes = 220 * 2 ** (torch.randint(0, 12, (2 * seconds,), generator=generator) / 12)
    f = notes.repeat_interleave(SR // 2)
    x = 0.3 * torch.sin(2 * torch.pi * torch.cumsum(f, 0) / SR)
    return x + 0.01 * torch.randn(len(x), generator=generator)


# a match is exact to within one feature hop, then may move by up to refine / 2
# samples to line up the boundary samples
TOL = 64 + 128

FEATURES = [STFTFeature(n_fft=256, hop_length=64, context=0.5), ChromaFeature(n_fft=256, hop_length=64, context=1.0)]


def test_finds_repeat(generator):
    x = _music(generator)
    x[12 * SR : 16 * SR] = x[3 * SR : 7 * SR]
    out = waveform_search(x[None], 13 * SR, 14 * SR, SR, n=3, search_rate=SR, features=FEATURES)
    assert abs(out.starts[0].item() - 4 * SR) < TOL
    assert (out.scores[1:] <= out.scores[:-1]).all()


def test_external_source(generator):
    x, other = _music(generator), _music(generator)
    other[5 * SR : 9 * SR] = x[9 * SR : 13 * SR]
    out = waveform_search(x[None], 10 * SR, 12 * SR, SR, source=other[None], search_rate=SR, features=FEATURES)
    assert abs(out.starts[0].item() - 6 * SR) < TOL


def test_fill_and_crossfade(generator):
    x = _music(generator)[None]
    a, b = 8 * SR, 10 * SR
    out = waveform_search(x, a, b, SR, n=2, search_rate=SR, features=FEATURES, crossfade=0.05)
    k = int(0.05 * SR)
    for sig, s in zip(out.signals, out.starts.tolist()):
        assert not (s < b and s + (b - a) > a)
        torch.testing.assert_close(sig[:, : a - k], x[:, : a - k])
        torch.testing.assert_close(sig[:, b + k :], x[:, b + k :])
        torch.testing.assert_close(sig[:, a:b], x[:, s : s + b - a])


def test_chroma_filterbank():
    fb = chroma_filterbank(16000, 513)
    assert fb.shape == (513, 12) and (fb >= 0).all()
    # a bin near A4 peaks at chroma 9 when starting at C
    k = round(440 / 8000 * 512)
    assert fb[k].argmax() == 9


def test_resampled_search(generator):
    x = _music(generator, 12)
    x[8 * SR : 10 * SR] = x[2 * SR : 4 * SR]
    out = waveform_search(x[None], 8 * SR + SR // 2, 9 * SR + SR // 2, SR, n=1, search_rate=4000,
                          features=[STFTFeature(n_fft=128, hop_length=32, context=0.5)])
    assert abs(out.starts[0].item() - 2 * SR - SR // 2) < TOL


def test_inpainter_uses_waveform_search(generator):
    prior = DenoiserPrior(gaussian_denoiser(0.0, 0.3), SR, 4 * SR, sigma_max=5.0)
    x = _music(generator)[None]
    out = Inpainter(prior, buffer=0).inpaint(x, SR, (9.0, 10.0), method="search", raw=True)
    assert out.candidates.signals.shape == (1, 1, 4 * SR)


def test_given_starts(generator):
    x = _music(generator)
    out = waveform_search(x[None], 13 * SR, 14 * SR, SR, search_rate=SR, features=FEATURES,
                          starts=[5 * SR, 3 * SR], refine=0)
    assert sorted(out.starts.tolist()) == [3 * SR, 5 * SR]
    assert (out.scores[1:] <= out.scores[:-1]).all()


def test_spacing(generator):
    x = _music(generator)
    out = waveform_search(x[None], 13 * SR, 14 * SR, SR, n=4, search_rate=SR, features=FEATURES,
                          spacing=SR, refine=0)
    s = sorted(out.starts.tolist())
    assert all(b - a >= SR - 64 for a, b in zip(s, s[1:]))


def test_band_feature_finds_repeat(generator):
    x = _music(generator)
    x[12 * SR : 16 * SR] = x[3 * SR : 7 * SR]
    bands = [BandFeature(n_fft=256, hop_length=64, context=1.0)]
    out = waveform_search(x[None], 13 * SR, 14 * SR, SR, n=3, search_rate=SR, features=bands)
    assert abs(out.starts[0].item() - 4 * SR) < TOL


def test_band_filterbank():
    fb = band_filterbank(16000, 513, n_bands=24)
    assert fb.shape == (513, 24)
    assert (fb.sum(0) >= 1).all() and fb[-1].sum() == 0
