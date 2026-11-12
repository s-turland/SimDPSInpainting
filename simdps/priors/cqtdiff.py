"""Waveform-domain piano model of the paper (Sec. 4.2).

A CQT U-Net trained on mono MAESTRO recordings at 44.1 kHz under the EDM
preconditioning. The network is in ``simdps/_vendor/cqtdiff``.
"""

import torch

from ..config import merge
from .base import EDMPrior

# Network of the paper's piano model. Override with the prior's ``network`` argument.
NETWORK = dict(
    representation="waveform",
    use_norm=True,
    depth=9,
    emb_dim=256,
    Ns=[32, 32, 64, 64, 128, 128, 256, 256, 256],
    attention_layers=[0, 0, 0, 0, 0, 0, 0, 0, 0],
    checkpointing=[False, False, False, False, False, False, False, False, False],
    Stime=[2, 1, 2, 2, 2, 1, 2, 2],
    Sfreq=[1, 2, 1, 1, 1, 2, 1, 1],
    F_dim=[32, 64, 48, 64, 80, 96, 56, 64, 72],
    F_bins_per_oct=[8, 8, 8, 16, 16, 16, 16, 32, 32],
    num_dils=[2, 2, 3, 4, 5, 5, 4, 5, 5],
    sample_rate=44100,
    audio_len=262144,
    cqt=dict(window="hann", num_octs=[3, 4, 2], bins_per_oct=[8, 16, 32]),
    bottleneck_type="res_dil_convs",
    num_bottleneck_layers=1,
    attention_dict=dict(
        num_heads=8, attn_dropout=0.0, bias_qkv=False, N=0, rel_pos_num_buckets=32,
        rel_pos_max_distance=64, use_rel_pos=True, Nproj=8,
    ),
)


class _Attributes(dict):
    # the network reads its nested settings as attributes
    __getattr__ = dict.get


class CQTDiff(EDMPrior):
    """Piano model of the paper, loaded from a checkpoint.

    Parameters
    ----------
    checkpoint : str
        Path to the trained weights: a state dict, or a training checkpoint
        with ``"ema"``, ``"ema_model"`` or ``"network"``.
    device : str or torch.device
    dtype : torch.dtype, optional
        Ignored: the transform runs in float32.
    network : dict, optional
        Changes to :data:`NETWORK`.
    sigma_data : float
        Standard deviation of the training data.
    """

    hop_length = 1
    sample_rate = 44100
    length = 262144
    sigma_min = 1e-5
    sigma_max = 8.0

    def __init__(self, checkpoint=None, device="cpu", dtype=None, network=None, sigma_data=0.063):
        if checkpoint is None:
            raise ValueError("this model needs its weights: set prior.checkpoint to the path of the checkpoint")
        settings = merge(NETWORK, network or {})
        super().__init__(build_network(settings, device), sigma_data, settings["sample_rate"], settings["audio_len"])
        self._device = torch.device(device)
        missing, unexpected = self.net.load_state_dict(read_weights(checkpoint, device), strict=False)
        if missing or unexpected:
            raise ValueError(f"{checkpoint} does not fit the network: {len(missing)} weights are missing, "
                             f"{len(unexpected)} unexpected; prior.network sets its shape")
        self.to(device).eval().requires_grad_(False)

    @property
    def device(self):
        return self._device

    def encode(self, audio):
        return audio.mean(1, keepdim=True)


def build_network(settings, device="cpu"):
    """Build the network from ``settings``, with random weights."""
    from .._vendor.cqtdiff.hd_unet_octCQT import HD_Unet_octCQT

    settings = dict(settings)
    settings["cqt"] = _Attributes(settings["cqt"])
    settings["attention_dict"] = _Attributes(settings["attention_dict"])
    return HD_Unet_octCQT(device=torch.device(device), **settings)


def read_weights(path, device="cpu"):
    """Read a state dict from a checkpoint, preferring the EMA weights.

    Training checkpoints are unpickled, so only load files you trust.
    """
    state = torch.load(path, map_location=device, weights_only=False)
    for key in ("ema", "ema_model", "network"):
        if isinstance(state, dict) and key in state:
            state = state[key]
            break
    if isinstance(state, list):
        state = state[0]
    return {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in state.items()}
