import pytest
import torch
from test_inpainter import SR, toy_prior

import simdps
from simdps import cli


def test_seconds():
    assert cli.seconds("12.5") == 12.5
    assert cli.seconds("1:02.5") == 62.5
    assert cli.seconds("1:00:00") == 3600


@pytest.fixture
def toy(monkeypatch):
    monkeypatch.setattr(simdps, "load", lambda *a, **k: simdps.Inpainter(toy_prior(), buffer=4))


def test_inpaint_writes_files(toy, tmp_path, generator):
    src = tmp_path / "take.wav"
    simdps.save_audio(src, torch.randn(2, 3 * SR, generator=generator) * 0.1, SR)
    cli.main(["inpaint", str(src), "--gap", "1.5", "1.55", "--steps", "3", "--samples", "2", "--seed", "1"])
    for i in (1, 2):
        y, sr = simdps.load_audio(tmp_path / f"take_simdps_{i}.wav")
        assert sr == SR and y.shape == (2, 3 * SR)


def test_inpaint_with_source(toy, tmp_path, generator):
    take, other = tmp_path / "take.flac", tmp_path / "other.wav"
    simdps.save_audio(take, torch.randn(1, 3 * SR, generator=generator) * 0.1, SR)
    simdps.save_audio(other, torch.randn(1, 2 * SR, generator=generator) * 0.1, SR)
    out = tmp_path / "fixed.flac"
    cli.main(["inpaint", str(take), "--gap", "0:01.5", "0:01.55", "--source", str(other),
              "--method", "solo", "--steps", "3", "-o", str(out)])
    assert simdps.load_audio(out)[0].shape == (1, 3 * SR)


def test_search_lists_matches(toy, tmp_path, generator, capsys):
    take = tmp_path / "take.wav"
    simdps.save_audio(take, torch.randn(1, 3 * SR, generator=generator) * 0.1, SR)
    cli.main(["inpaint", str(take), "--gap", "1.5", "1.55", "--method", "search", "--samples", "3", "--seed", "0"])
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 3 and all(line.endswith(" s)") for line in lines)


def test_guide_candidates_and_folder(toy, tmp_path, generator):
    take, lib = tmp_path / "take.wav", tmp_path / "lib"
    lib.mkdir()
    simdps.save_audio(take, torch.randn(1, 3 * SR, generator=generator) * 0.1, SR)
    simdps.save_audio(lib / "a.wav", torch.randn(1, 2 * SR, generator=generator) * 0.1, SR)
    base = ["inpaint", str(take), "--gap", "1.5", "1.55", "--steps", "2"]
    cli.main(base + ["--guide", "0:00.8", "-o", str(tmp_path / "g.wav")])
    cli.main(base + ["--candidates", "2", "1", "-o", str(tmp_path / "c.wav")])
    cli.main(base + ["--source", str(lib), "-o", str(tmp_path / "f.wav")])
    for name in "gcf":
        assert simdps.load_audio(tmp_path / f"{name}.wav")[0].shape == (1, 3 * SR)


def test_match_by_bands(toy, tmp_path, generator):
    take = tmp_path / "take.wav"
    simdps.save_audio(take, torch.randn(1, 15 * SR, generator=generator) * 0.1, SR)
    cli.main(["inpaint", str(take), "--gap", "7", "7.05", "--match", "bands", "--steps", "2"])
    assert simdps.load_audio(tmp_path / "take_simdps.wav")[0].shape == (1, 15 * SR)


def test_config_and_set(toy, tmp_path, generator):
    take = tmp_path / "take.wav"
    simdps.save_audio(take, torch.randn(1, 3 * SR, generator=generator) * 0.1, SR)
    out = tmp_path / "o.wav"
    cli.main(["inpaint", str(take), "--gap", "1.5", "1.55", "--config", "warm_start_fast",
              "--set", "sampler.steps=2", "candidates.weight=0.2", "-o", str(out)])
    assert simdps.load_audio(out)[0].shape == (1, 3 * SR)


@pytest.mark.parametrize("extra,method", [(["--guide", "0.8"], "solo"), (["--candidates", "1"], "solo"),
                                          (["--candidates", "2", "1"], None), (["--guide", "0.8", "--method", "multi"],
                                                                               "multi")])
def test_single_guide_uses_solo(toy, tmp_path, generator, monkeypatch, extra, method):
    seen = []
    inpaint = simdps.Inpainter.inpaint
    record = lambda self, *a, **k: seen.append(k["method"]) or inpaint(self, *a, **k)
    monkeypatch.setattr(simdps.Inpainter, "inpaint", record)
    take = tmp_path / "take.wav"
    simdps.save_audio(take, torch.randn(1, 3 * SR, generator=generator) * 0.1, SR)
    cli.main(["inpaint", str(take), "--gap", "1.5", "1.55", "--steps", "2", *extra])
    assert seen == [method]
