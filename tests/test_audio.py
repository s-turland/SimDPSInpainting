import pytest
import torch

from simdps.audio import load_audio, save_audio


def test_wav_keeps_overs(tmp_path):
    x = torch.linspace(-1.5, 1.5, 1000)[None]
    save_audio(tmp_path / "a.wav", x, 8000)
    y, sr = load_audio(tmp_path / "a.wav")
    assert sr == 8000 and torch.equal(x, y)


def test_flac_warns_on_clipping(tmp_path):
    with pytest.warns(UserWarning, match="clipping"):
        save_audio(tmp_path / "a.flac", torch.full((1, 100), 1.2), 8000)


def test_resample_on_load(tmp_path):
    save_audio(tmp_path / "a.wav", torch.zeros(2, 8000), 8000)
    y, sr = load_audio(tmp_path / "a.wav", sample_rate=16000)
    assert y.shape == (2, 16000) and sr == 16000
