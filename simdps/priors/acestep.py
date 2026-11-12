"""ACE-Step 1.5 (Gong et al., 2026) as an unconditional latent prior.

The diffusion transformer is run without text or lyrics, with the null
condition embedding, on latents of the Oobleck VAE (64 channels, 25 Hz, 48 kHz
stereo audio).
"""

import logging
import os
import warnings
from contextlib import contextmanager, nullcontext

import torch

from .base import FlowPrior

DIT_REPO = "ACE-Step/acestep-v15-base"
DIT_REVISION = "e432212fec32b8965a14ffa57ae653438d6abd14"
VAE_REPO = "ACE-Step/Ace-Step1.5"
VAE_REVISION = "19671f406d603126926c1b7e2adc169acbcade22"

# flow-matching times the turbo model was distilled on (shift 3, 8 steps),
# without t = 1, as noise levels sigma = t / (1 - t)
TURBO_TIMES = [0.9545454545454546, 0.9, 0.8333333333333334, 0.75, 0.6428571428571429, 0.5, 0.3]
TURBO_SIGMAS = [t / (1 - t) for t in TURBO_TIMES]


class ACEStep(FlowPrior):
    """ACE-Step 1.5 base model with its VAE.

    Parameters
    ----------
    model : {"base", "turbo"}
        ``"turbo"`` is the distilled model, sampled on the 7 noise levels it
        was trained for (see ``sigmas``).
    device : str or torch.device
    dtype : torch.dtype
        Precision of the transformer. ``torch.bfloat16`` halves the memory
        (about 5 GB with gradients); the VAE always runs in float32.
    sample_latent : bool
        Encode by sampling the VAE posterior, as in the paper, rather than
        taking its mean.
    cache_dir : str, optional
        Where to keep the downloaded weights.
    """

    sample_rate = 48000
    hop_length = 1920
    length = 375
    pad_frames = 128
    sigma_min = 1e-3
    sigma_max = 20.0
    cache_id = f"{VAE_REPO}@{VAE_REVISION}"

    def __init__(self, model="base", device="cpu", dtype=torch.float32, sample_latent=True, cache_dir=None):
        if model not in ("base", "turbo"):
            raise ValueError(f"unknown ACE-Step model '{model}'")
        with _quiet():
            super().__init__(_VelocityNet(model, device, dtype, cache_dir), self.sample_rate, self.length)
            self.vae = _load_vae(cache_dir).to(device).eval().requires_grad_(False)
        self.model = model
        self.sigmas = TURBO_SIGMAS if model == "turbo" else None
        self.sample_latent = sample_latent
        self.encode_chunk = 256
        self.encode_context = 32
        self.decode_chunk = None
        self.decode_context = 32

    @property
    def device(self):
        return self.net.device

    def denoise(self, x, sigma):
        return super().denoise(x, sigma).to(x.dtype)

    @torch.no_grad()
    def encode(self, audio):
        """``(batch, channels, samples)`` audio at 48 kHz to ``(batch, 64, frames)``."""
        from diffusers.models.autoencoders.autoencoder_oobleck import OobleckDiagonalGaussianDistribution

        if audio.shape[1] == 1:
            audio = audio.repeat(1, 2, 1)
        with _conv_backend():
            h = self._encoder(audio.to(self.device, torch.float32))
        dist = OobleckDiagonalGaussianDistribution(h)
        z = dist.sample() if self.sample_latent else dist.mean
        _release(h)
        return z

    def decode(self, z):
        z = z.to(torch.float32)
        with _conv_backend():
            if self.decode_chunk is None or z.shape[-1] <= self.decode_chunk + 2 * self.decode_context:
                y = self.vae.decode(z).sample
            else:
                y = self._decode_chunked(z)
        _release(y)
        return y

    def _decode_chunked(self, z):
        n, chunk, ctx, hop = z.shape[-1], self.decode_chunk, self.decode_context, self.hop_length
        parts = []
        for start in range(0, n, chunk):
            lo, hi = max(start - ctx, 0), min(start + chunk + ctx, n)
            y = self.vae.decode(z[..., lo:hi]).sample
            off = (start - lo) * hop
            parts.append(y[..., off : off + min(chunk, n - start) * hop])
        return torch.cat(parts, dim=-1)

    def _encoder(self, x):
        # encode long inputs in chunks, with enough context on each side for
        # every kept frame to match a one-shot encode
        hop = self.hop_length
        chunk, ctx = self.encode_chunk * hop, self.encode_context * hop
        n = x.shape[-1]
        if n <= chunk + 2 * ctx:
            return self.vae.encoder(x)
        parts = []
        for start in range(0, n, chunk):
            lo = max(start - ctx, 0)
            last = start + chunk + ctx >= n
            h = self.vae.encoder(x[..., lo : n if last else start + chunk + ctx])
            off = (start - lo) // hop
            parts.append(h[..., off:] if last else h[..., off : off + self.encode_chunk])
            if last:
                break
        return torch.cat(parts, dim=-1)


class _VelocityNet(torch.nn.Module):
    def __init__(self, model, device, dtype, cache_dir):
        super().__init__()
        from huggingface_hub import hf_hub_download
        from safetensors import safe_open

        from .._vendor.acestep.configuration_acestep_v15 import AceStepConfig
        from .._vendor.acestep.modeling import AceStepDiTModel, Qwen3RotaryEmbedding

        def get(f):
            if model == "base":
                return hf_hub_download(DIT_REPO, f, revision=DIT_REVISION, cache_dir=cache_dir)
            return hf_hub_download(VAE_REPO, "acestep-v15-turbo/" + f, revision=VAE_REVISION, cache_dir=cache_dir)

        config = AceStepConfig.from_json_file(get("config.json"))
        config._attn_implementation = "sdpa"
        # build on the meta device and fill tensor by tensor on the target device,
        # so the float32 weights are never all in RAM
        with torch.device("meta"):
            self.dit = AceStepDiTModel(config)
        state, null = {}, None
        with safe_open(get("model.safetensors"), framework="pt", device=str(device)) as f:
            for k in f.keys():
                if k.startswith("decoder."):
                    state[k[len("decoder.") :]] = f.get_tensor(k).to(dtype)
                elif k == "null_condition_emb":
                    null = f.get_tensor(k).to(dtype)
        self.dit.load_state_dict(state, assign=True)
        self.dit.rotary_emb = Qwen3RotaryEmbedding(config).to(device, dtype)
        self.register_buffer("null", null)
        silence = torch.load(get("silence_latent.pt"), map_location=device, weights_only=True)
        self.register_buffer("silence", silence.transpose(1, 2).to(dtype))
        self.eval().requires_grad_(False)

    @property
    def device(self):
        return self.null.device

    def forward(self, x, t):
        # x (batch, 64, frames) -> the transformer's (batch, frames, 64)
        dtype = self.null.dtype
        x = x.transpose(1, 2).to(dtype)
        b, n, _ = x.shape
        src = self.silence[0, :n].to(dtype).expand(b, -1, -1)
        context = torch.cat([src, torch.ones_like(src)], dim=-1)
        cond = self.null.expand(b, -1, -1)
        t = t.to(dtype) * x.new_ones(b)
        v = self.dit(
            hidden_states=x,
            timestep=t,
            timestep_r=t,
            attention_mask=x.new_ones(b, n),
            encoder_hidden_states=cond,
            encoder_attention_mask=torch.ones(cond.shape[:-1], device=x.device, dtype=torch.bool),
            context_latents=context,
            use_cache=False,
        )[0]
        return v.transpose(1, 2).float()


@contextmanager
def _quiet():
    # silence deprecation and download notices from transformers, diffusers and the hub
    loggers = [logging.getLogger(n) for n in ("transformers", "diffusers", "huggingface_hub")]
    levels = [lg.level for lg in loggers]
    for lg in loggers:
        lg.setLevel(logging.ERROR)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            warnings.filterwarnings("ignore", message=".*local_dir_use_symlinks.*")
            warnings.filterwarnings("ignore", message=".*unauthenticated.*")
            yield
    finally:
        for lg, level in zip(loggers, levels):
            lg.setLevel(level)


def _load_vae(cache_dir):
    from diffusers.models import AutoencoderOobleck

    return AutoencoderOobleck.from_pretrained(
        VAE_REPO, subfolder="vae", revision=VAE_REVISION, cache_dir=cache_dir
    )


def _release(t):
    # free cached blocks after the VAE's large activations, otherwise the
    # transformer spills into shared memory on small GPUs
    if t.is_cuda:
        torch.cuda.empty_cache()


def _conv_backend():
    # MIOpen falls back to very slow kernels for the VAE's long 1D convolutions
    # on ROCm; im2col + GEMM is much faster. SIMDPS_VAE_MIOPEN=1 keeps MIOpen.
    if torch.version.hip is not None and os.environ.get("SIMDPS_VAE_MIOPEN") != "1":
        return torch.backends.cudnn.flags(enabled=False)
    return nullcontext()
