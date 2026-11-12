import pytest
import torch


def gaussian_denoiser(mean, std):
    """Exact E[x0 | x] for x0 ~ N(mean, std^2) and x = x0 + sigma * n."""

    def denoise(x, sigma):
        s2 = sigma.view(-1, *([1] * (x.ndim - 1))) ** 2
        return (std**2 * x + s2 * mean) / (std**2 + s2)

    return denoise


@pytest.fixture
def generator():
    return torch.Generator().manual_seed(0)


@pytest.fixture(autouse=True)
def _cache(tmp_path, monkeypatch):
    monkeypatch.setenv("SIMDPS_CACHE", str(tmp_path / "cache"))
