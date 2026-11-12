import pytest
import torch

from simdps.search import interpolate, latent_search

C, L = 6, 400
START, END = 200, 240


def _planted(generator, at=60, context=5):
    z = torch.randn(C, L, generator=generator)
    gap = END - START
    z[:, at - context : at] = z[:, START - context : START]
    z[:, at + gap : at + gap + context] = z[:, END : END + context]
    return z


@pytest.mark.parametrize("metric", ["l2", "l1", "cosine", "pearson"])
@pytest.mark.parametrize("ramp", [False, True])
def test_finds_planted_context(metric, ramp, generator):
    z = _planted(generator)
    out = latent_search(z, START, END, n=5, metric=metric, ramp=ramp)
    assert out.starts[0] == 60
    assert out.scores[0] == pytest.approx(1.0, abs=1e-5)
    assert (out.scores[1:] <= out.scores[:-1]).all()


def test_never_overlaps_gap(generator):
    z = torch.randn(C, L, generator=generator)
    out = latent_search(z, START, END, n=50, crossfade=5)
    s = out.starts
    assert ((s + (END - START) + 5 <= START) | (s - 5 >= END)).all()


def test_external_source(generator):
    target = torch.randn(C, L, generator=generator)
    source = torch.randn(C, 300, generator=generator)
    source[:, START - 5 : START] = target[:, START - 5 : START]
    source[:, END : END + 5] = target[:, END : END + 5]
    out = latent_search(target, START, END, source=source, n=3)
    assert out.starts[0] == START
    torch.testing.assert_close(out.signals[0, :, START + 5 : END - 5], source[:, START + 5 : END - 5])


def test_fill_leaves_context(generator):
    z = torch.randn(C, L, generator=generator)
    out = latent_search(z, START, END, n=4, crossfade=5)
    for sig, s in zip(out.signals, out.starts.tolist()):
        torch.testing.assert_close(sig[:, : START - 5], z[:, : START - 5])
        torch.testing.assert_close(sig[:, END + 5 :], z[:, END + 5 :])
        torch.testing.assert_close(sig[:, START:END], z[:, s : s + END - START])


def test_crop(generator):
    out = latent_search(torch.randn(C, L, generator=generator), START, END, n=2)
    assert out.crop(150, 120).signals.shape == (2, C, 120)
    assert len(out) == 2


def test_too_short():
    with pytest.raises(ValueError):
        latent_search(torch.randn(C, 50), 10, 40)


@pytest.mark.parametrize("mode", ["lerp", "slerp"])
def test_interpolate_endpoints(mode, generator):
    a = torch.randn(C, 1, generator=generator).expand(C, 5)
    b = torch.randn(C, 1, generator=generator).expand(C, 5)
    out = interpolate(a, b, torch.linspace(0, 1, 5), mode)
    torch.testing.assert_close(out[:, 0], a[:, 0], atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(out[:, -1], b[:, -1], atol=1e-3, rtol=1e-3)


def test_slerp_keeps_norm_of_unit_vectors():
    a = torch.tensor([[1.0], [0.0]]).expand(2, 9)
    b = torch.tensor([[0.0], [1.0]]).expand(2, 9)
    out = interpolate(a, b, torch.linspace(0, 1, 9), "slerp")
    torch.testing.assert_close(out.norm(dim=0), torch.ones(9), atol=1e-3, rtol=1e-3)


def test_given_starts(generator):
    z = _planted(generator)
    found = latent_search(z, START, END, n=5)
    out = latent_search(z, START, END, starts=[found.starts[2], 60, found.starts[1]])
    assert out.starts.tolist() == found.starts[:3].tolist()
    torch.testing.assert_close(out.scores, found.scores[:3])
    torch.testing.assert_close(out.signals, found.signals[:3])


def test_given_starts_out_of_range(generator):
    with pytest.raises(ValueError):
        latent_search(torch.randn(C, L, generator=generator), START, END, starts=[L - 10])


def test_spacing(generator):
    z = _planted(generator)
    close = latent_search(z, START, END, n=5)
    apart = latent_search(z, START, END, n=5, spacing=END - START)
    s = apart.starts.tolist()
    assert s[0] == 60 and all(abs(a - b) >= END - START for i, a in enumerate(s) for b in s[:i])
    assert (apart.scores[1:] <= apart.scores[:-1]).all()
    assert (close.starts[1:] - close.starts[0]).abs().min() < END - START
