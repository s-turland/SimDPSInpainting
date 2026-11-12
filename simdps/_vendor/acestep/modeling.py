# Copyright 2025 The ACESTEO Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified for SimDPS: only the diffusion transformer (AceStepDiTModel) and
# its dependencies are kept, unchanged, from modeling_acestep_v15_base.py of
# https://huggingface.co/ACE-Step/acestep-v15-base.

import math
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers.cache_utils import Cache, DynamicCache, EncoderDecoderCache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3MLP,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from transformers.processing_utils import Unpack
from transformers.utils import auto_docstring, can_return_tuple, logging

from .configuration_acestep_v15 import AceStepConfig

logger = logging.get_logger(__name__)



def create_4d_mask(
    seq_len: int,
    dtype: torch.dtype,
    device: torch.device,
    attention_mask: Optional[torch.Tensor] = None,  # [Batch, Seq_Len]
    sliding_window: Optional[int] = None,
    is_sliding_window: bool = False,
    is_causal: bool = True,
) -> torch.Tensor:
    """
    General 4D Attention Mask generator compatible with CPU/Mac/SDPA and Eager mode.
    Supports use cases:
    1. Causal Full: is_causal=True, is_sliding_window=False (standard GPT)
    2. Causal Sliding: is_causal=True, is_sliding_window=True (Mistral/Qwen local window)
    3. Bidirectional Full: is_causal=False, is_sliding_window=False (BERT/Encoder)
    4. Bidirectional Sliding: is_causal=False, is_sliding_window=True (Longformer local)

    Returns:
        [Batch, 1, Seq_Len, Seq_Len] additive mask (0.0 for keep, -inf for mask)
    """
    # ------------------------------------------------------
    # 1. Construct basic geometry mask [Seq_Len, Seq_Len]
    # ------------------------------------------------------

    # Build index matrices
    # i (Query): [0, 1, ..., L-1]
    # j (Key):   [0, 1, ..., L-1]
    indices = torch.arange(seq_len, device=device)
    # diff = i - j
    diff = indices.unsqueeze(1) - indices.unsqueeze(0)

    # Initialize all True (all positions visible)
    valid_mask = torch.ones((seq_len, seq_len), device=device, dtype=torch.bool)

    # (A) Handle causality (Causal)
    if is_causal:
        # i >= j  =>  diff >= 0
        valid_mask = valid_mask & (diff >= 0)

    # (B) Handle sliding window
    if is_sliding_window and sliding_window is not None:
        if is_causal:
            # Causal sliding: only attend to past window steps
            # i - j <= window  =>  diff <= window
            # (diff >= 0 already handled above)
            valid_mask = valid_mask & (diff <= sliding_window)
        else:
            # Bidirectional sliding: attend past and future window steps
            # |i - j| <= window  =>  abs(diff) <= sliding_window
            valid_mask = valid_mask & (torch.abs(diff) <= sliding_window)

    # Expand dimensions to [1, 1, Seq_Len, Seq_Len] for broadcasting
    valid_mask = valid_mask.unsqueeze(0).unsqueeze(0)

    # ------------------------------------------------------
    # 2. Apply padding mask (Key Masking)
    # ------------------------------------------------------
    if attention_mask is not None:
        # attention_mask shape: [Batch, Seq_Len] (1=valid, 0=padding)
        # We want to mask out invalid keys (columns)
        # Expand shape: [Batch, 1, 1, Seq_Len]
        padding_mask_4d = attention_mask.view(
            attention_mask.shape[0], 1, 1, seq_len
        ).to(torch.bool)

        # Broadcasting: Geometry Mask [1, 1, L, L] & Padding Mask [B, 1, 1, L]
        # Result shape: [B, 1, L, L]
        valid_mask = valid_mask & padding_mask_4d

    # ------------------------------------------------------
    # 3. Convert to additive mask
    # ------------------------------------------------------
    # Get the minimal value for current dtype
    min_dtype = torch.finfo(dtype).min

    # Create result tensor filled with -inf by default
    mask_tensor = torch.full(valid_mask.shape, min_dtype, dtype=dtype, device=device)

    # Set valid positions to 0.0
    mask_tensor.masked_fill_(valid_mask, 0.0)

    return mask_tensor



class TimestepEmbedding(nn.Module):
    """
    Timestep embedding module for diffusion models.

    Converts timestep values into high-dimensional embeddings using sinusoidal
    positional encoding, followed by MLP layers. Used for conditioning diffusion
    models on timestep information.
    """

    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        scale: float = 1000,
    ):
        super().__init__()

        self.linear_1 = nn.Linear(in_channels, time_embed_dim, bias=True)
        self.act1 = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim, bias=True)
        self.in_channels = in_channels

        self.act2 = nn.SiLU()
        self.time_proj = nn.Linear(time_embed_dim, time_embed_dim * 6)
        self.scale = scale

    def timestep_embedding(self, t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.

        Args:
            t: A 1-D tensor of N indices, one per batch element. These may be fractional.
            dim: The dimension of the output embeddings.
            max_period: Controls the minimum frequency of the embeddings.

        Returns:
            An (N, D) tensor of positional embeddings.
        """
        t = t * self.scale
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.in_channels)
        temb = self.linear_1(t_freq.to(t.dtype))
        temb = self.act1(temb)
        temb = self.linear_2(temb)
        timestep_proj = self.time_proj(self.act2(temb)).unflatten(1, (6, -1))
        return temb, timestep_proj



class AceStepAttention(nn.Module):
    """
    Multi-headed attention module for AceStep model.

    Implements the attention mechanism from 'Attention Is All You Need' paper,
    with support for both self-attention and cross-attention modes. Uses RMSNorm
    for query and key normalization, and supports sliding window attention for
    efficient long-sequence processing.
    """

    def __init__(
        self,
        config: AceStepConfig,
        layer_idx: int,
        is_cross_attention: bool = False,
        is_causal: bool = False,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        if is_cross_attention:
            is_causal = False
        self.is_causal = is_causal
        self.is_cross_attention = is_cross_attention

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        # Apply RMS normalization only on the head dimension (unlike OLMo)
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx]
        self.sliding_window = (
            config.sliding_window
            if config.layer_types[layer_idx] == "sliding_attention"
            else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # Project and normalize query states
        query_states = self.q_norm(
            self.q_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)

        # Determine if this is cross-attention (requires encoder_hidden_states)
        is_cross_attention = (
            self.is_cross_attention and encoder_hidden_states is not None
        )

        # Cross-attention path: attend to encoder hidden states
        if is_cross_attention:
            encoder_hidden_shape = (
                *encoder_hidden_states.shape[:-1],
                -1,
                self.head_dim,
            )
            if past_key_value is not None:
                is_updated = past_key_value.is_updated.get(self.layer_idx)
                # After the first generated token, we can reuse all key/value states from cache
                curr_past_key_value = past_key_value.cross_attention_cache

                # Conditions for calculating key and value states
                if not is_updated:
                    # Compute and cache K/V for the first time
                    key_states = self.k_norm(
                        self.k_proj(encoder_hidden_states).view(encoder_hidden_shape)
                    ).transpose(1, 2)
                    value_states = (
                        self.v_proj(encoder_hidden_states)
                        .view(encoder_hidden_shape)
                        .transpose(1, 2)
                    )
                    # Update cache: save all key/value states to cache for fast auto-regressive generation
                    key_states, value_states = curr_past_key_value.update(
                        key_states, value_states, self.layer_idx
                    )
                    # Set flag that this layer's cross-attention cache is updated
                    past_key_value.is_updated[self.layer_idx] = True
                else:
                    # Reuse cached key/value states for subsequent tokens
                    key_states = curr_past_key_value.layers[self.layer_idx].keys
                    value_states = curr_past_key_value.layers[self.layer_idx].values
            else:
                # No cache used, compute K/V directly
                key_states = self.k_norm(
                    self.k_proj(encoder_hidden_states).view(encoder_hidden_shape)
                ).transpose(1, 2)
                value_states = (
                    self.v_proj(encoder_hidden_states)
                    .view(encoder_hidden_shape)
                    .transpose(1, 2)
                )

        # Self-attention path: attend to the same sequence
        else:
            # Project and normalize key/value states for self-attention
            key_states = self.k_norm(
                self.k_proj(hidden_states).view(hidden_shape)
            ).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            # Apply rotary position embeddings (RoPE) if provided
            if position_embeddings is not None:
                cos, sin = position_embeddings
                query_states, key_states = apply_rotary_pos_emb(
                    query_states, key_states, cos, sin
                )

            # Update cache for auto-regressive generation
            if past_key_value is not None:
                # Sin and cos are specific to RoPE models; cache_position needed for the static cache
                cache_kwargs = {
                    "sin": sin,
                    "cos": cos,
                    "cache_position": cache_position,
                }
                key_states, value_states = past_key_value.update(
                    key_states, value_states, self.layer_idx, cache_kwargs
                )

        attention_interface: Callable = eager_attention_forward
        if is_cross_attention and output_attentions:
            attention_interface: Callable = eager_attention_forward
        elif self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation
            ]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            sliding_window=self.sliding_window if not self.is_cross_attention else None,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights



class AceStepDiTLayer(GradientCheckpointingLayer):
    """
    DiT (Diffusion Transformer) layer for AceStep model.

    Implements a transformer layer with three main components:
    1. Self-attention with adaptive layer norm (AdaLN)
    2. Cross-attention (optional) for conditioning on encoder outputs
    3. Feed-forward MLP with adaptive layer norm

    Uses scale-shift modulation from timestep embeddings for adaptive normalization.
    """

    def __init__(
        self, config: AceStepConfig, layer_idx: int, use_cross_attention: bool = True
    ):
        super().__init__()

        # 1. Self-attention sub-layer with adaptive normalization
        self.self_attn_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = AceStepAttention(config=config, layer_idx=layer_idx)

        # 2. Cross-attention sub-layer (optional, for encoder conditioning)
        self.use_cross_attention = use_cross_attention
        if self.use_cross_attention:
            self.cross_attn_norm = Qwen3RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.cross_attn = AceStepAttention(
                config=config, layer_idx=layer_idx, is_cross_attention=True
            )

        # 3. Feed-forward MLP sub-layer with adaptive normalization
        self.mlp_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = Qwen3MLP(config)

        # Scale-shift table for adaptive layer norm modulation (6 values: 3 for self-attn, 3 for MLP)
        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 6, config.hidden_size) / config.hidden_size**0.5
        )
        self.attention_type = config.layer_types[layer_idx]

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        temb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[EncoderDecoderCache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:

        # Extract scale-shift parameters for adaptive layer norm from timestep embeddings
        # 6 values: (shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa)
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            self.scale_shift_table + temb
        ).chunk(6, dim=1)

        # Step 1: Self-attention with adaptive layer norm (AdaLN)
        # Apply adaptive normalization: norm(x) * (1 + scale) + shift
        norm_hidden_states = (
            self.self_attn_norm(hidden_states) * (1 + scale_msa) + shift_msa
        ).type_as(hidden_states)
        attn_output, self_attn_weights = self.self_attn(
            hidden_states=norm_hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            use_cache=False,
            past_key_value=None,
            **kwargs,
        )
        # Apply gated residual connection: x = x + attn_output * gate
        hidden_states = (hidden_states + attn_output * gate_msa).type_as(hidden_states)

        # Step 2: Cross-attention (if enabled) for conditioning on encoder outputs
        if self.use_cross_attention:
            norm_hidden_states = self.cross_attn_norm(hidden_states).type_as(
                hidden_states
            )
            attn_output, cross_attn_weights = self.cross_attn(
                hidden_states=norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                **kwargs,
            )
            # Standard residual connection for cross-attention
            hidden_states = hidden_states + attn_output

        # Step 3: Feed-forward (MLP) with adaptive layer norm
        # Apply adaptive normalization for MLP: norm(x) * (1 + scale) + shift
        norm_hidden_states = (
            self.mlp_norm(hidden_states) * (1 + c_scale_msa) + c_shift_msa
        ).type_as(hidden_states)
        ff_output = self.mlp(norm_hidden_states)
        # Apply gated residual connection: x = x + mlp_output * gate
        hidden_states = (hidden_states + ff_output * c_gate_msa).type_as(hidden_states)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        return outputs



@auto_docstring
class AceStepPreTrainedModel(PreTrainedModel):
    config_class = AceStepConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["AceStepEncoderLayer", "AceStepDiTLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_3 = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        """
        Initialize weights for different module types.

        TODO: Support separate initialization for encoders and decoders.
        """
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, Qwen3RMSNorm):
            module.weight.data.fill_(1.0)



class Lambda(nn.Module):
    """
    Wrapper module for arbitrary lambda functions.

    Allows using lambda functions in nn.Sequential by wrapping them in a Module.
    Useful for simple transformations like transpose operations.
    """

    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)



class AceStepDiTModel(AceStepPreTrainedModel):
    """
    DiT (Diffusion Transformer) model for AceStep.

    Main diffusion model that generates audio latents conditioned on text, lyrics,
    and timbre. Uses patch-based processing with transformer layers, timestep
    conditioning, and cross-attention to encoder outputs.
    """

    def __init__(self, config: AceStepConfig):
        super().__init__(config)
        # Rotary position embeddings for transformer layers
        self.rotary_emb = Qwen3RotaryEmbedding(config)
        # Stack of DiT transformer layers
        self.layers = nn.ModuleList(
            [
                AceStepDiTLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        in_channels = config.in_channels
        inner_dim = config.hidden_size
        patch_size = config.patch_size
        self.patch_size = patch_size

        # Input projection: patch embedding using 1D convolution
        # Converts sequence into patches for efficient processing
        self.proj_in = nn.Sequential(
            Lambda(lambda x: x.transpose(1, 2)),  # [B, T, C] -> [B, C, T]
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=inner_dim,
                kernel_size=patch_size,
                stride=patch_size,
                padding=0,
            ),
            Lambda(
                lambda x: x.transpose(1, 2)
            ),  # [B, C, T//patch_size] -> [B, T//patch_size, C]
        )

        # Timestep embeddings for diffusion conditioning
        # Two embeddings: one for timestep t, one for timestep difference (t - r)
        self.time_embed = TimestepEmbedding(in_channels=256, time_embed_dim=inner_dim)
        self.time_embed_r = TimestepEmbedding(in_channels=256, time_embed_dim=inner_dim)

        # Project encoder hidden states to model dimension
        self.condition_embedder = nn.Linear(inner_dim, inner_dim, bias=True)

        # Output normalization and projection
        # Adaptive layer norm with scale-shift modulation, then de-patchify
        self.norm_out = Qwen3RMSNorm(inner_dim, eps=config.rms_norm_eps)
        self.proj_out = nn.Sequential(
            Lambda(
                lambda x: x.transpose(1, 2)
            ),  # [B, T//patch_size, inner_dim] -> [B, inner_dim, T//patch_size]
            nn.ConvTranspose1d(
                in_channels=inner_dim,
                out_channels=config.audio_acoustic_hidden_dim,
                kernel_size=patch_size,
                stride=patch_size,
                padding=0,
            ),
            Lambda(
                lambda x: x.transpose(1, 2)
            ),  # [B, out_channels, T] -> [B, T, out_channels]
        )
        # Scale-shift table for adaptive output normalization (2 values: shift, scale)
        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 2, inner_dim) / inner_dim**0.5
        )

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        timestep_r: torch.Tensor,
        attention_mask: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        context_latents: torch.Tensor,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        return_hidden_states: int = None,
        custom_layers_config: Optional[dict] = None,
        enable_early_exit: bool = False,
        past_key_values: Optional[EncoderDecoderCache] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ):

        use_cache = use_cache if use_cache is not None else self.config.use_cache

        # Disable cache during training or when gradient checkpointing is enabled
        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False
        if self.training:
            use_cache = False

        # Initialize cache if needed (only during inference for auto-regressive generation)
        if not self.training and use_cache and past_key_values is None:
            past_key_values = EncoderDecoderCache(DynamicCache(), DynamicCache())

        # Compute timestep embeddings for diffusion conditioning
        # Two embeddings: one for timestep t, one for timestep difference (t - r)
        temb_t, timestep_proj_t = self.time_embed(timestep)
        temb_r, timestep_proj_r = self.time_embed_r(timestep - timestep_r)
        # Combine embeddings
        temb = temb_t + temb_r
        timestep_proj = timestep_proj_t + timestep_proj_r

        # Concatenate context latents (source latents + chunk masks) with hidden states
        hidden_states = torch.cat([context_latents, hidden_states], dim=-1)
        # Record original sequence length for later restoration after padding
        original_seq_len = hidden_states.shape[1]
        # Apply padding if sequence length is not divisible by patch_size
        # This ensures proper patch extraction
        pad_length = 0
        if hidden_states.shape[1] % self.patch_size != 0:
            pad_length = self.patch_size - (hidden_states.shape[1] % self.patch_size)
            hidden_states = F.pad(
                hidden_states, (0, 0, 0, pad_length), mode="constant", value=0
            )

        # Project input to patches and project encoder states
        hidden_states = self.proj_in(hidden_states)
        encoder_hidden_states = self.condition_embedder(encoder_hidden_states)

        # Cache positions
        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + hidden_states.shape[1],
                device=hidden_states.device,
            )

        # Position IDs
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        seq_len = hidden_states.shape[1]
        encoder_seq_len = encoder_hidden_states.shape[1]
        dtype = hidden_states.dtype
        device = hidden_states.device

        # 判断是否使用 Flash Attention 2
        is_flash_attn = self.config._attn_implementation == "flash_attention_2"

        # 初始化 Mask 变量
        full_attn_mask = None
        sliding_attn_mask = None
        encoder_attention_mask = None
        attention_mask = None
        if is_flash_attn:
            # -------------------------------------------------------
            # 场景 A: Flash Attention 模式
            # -------------------------------------------------------
            # FA 不需要 4D Mask。
            # 如果有 padding mask (attention_mask [B, L])，直接传给它即可。
            # 如果没有 padding mask，传 None。
            # 滑动窗口逻辑由 Layer 内部传给 FA kernel 的 sliding_window 参数控制。
            full_attn_mask = attention_mask

            # 这里的逻辑是：如果配置启用了滑动窗口，FA 模式下我们也只需要传基础的 padding mask
            # Layer 会自己决定是否调用带 sliding window 的 kernel
            sliding_attn_mask = (
                attention_mask if self.config.use_sliding_window else None
            )

        else:
            # -------------------------------------------------------
            # 场景 B: CPU / Mac / SDPA (Eager 模式)
            # -------------------------------------------------------
            # 必须手动生成 4D Mask [B, 1, L, L]

            # 1. Full Attention (Bidirectional, Global)
            # 对应原来的 create_causal_mask + bidirectional
            full_attn_mask = create_4d_mask(
                seq_len=seq_len,
                dtype=dtype,
                device=device,
                attention_mask=attention_mask,  # [B, L]
                sliding_window=None,
                is_sliding_window=False,
                is_causal=False,  # <--- 关键：双向注意力
            )
            max_len = max(seq_len, encoder_seq_len)

            encoder_attention_mask = create_4d_mask(
                seq_len=max_len,
                dtype=dtype,
                device=device,
                attention_mask=attention_mask,  # [B, L]
                sliding_window=None,
                is_sliding_window=False,
                is_causal=False,  # <--- 关键：双向注意力
            )
            encoder_attention_mask = encoder_attention_mask[
                :, :, :seq_len, :encoder_seq_len
            ]
            # 2. Sliding Attention (Bidirectional, Local)
            # 对应原来的 create_sliding_window... + bidirectional
            if self.config.use_sliding_window:
                sliding_attn_mask = create_4d_mask(
                    seq_len=seq_len,
                    dtype=dtype,
                    device=device,
                    attention_mask=attention_mask,  # [B, L]
                    sliding_window=self.config.sliding_window,
                    is_sliding_window=True,  # <--- 开启滑动窗口
                    is_causal=False,  # <--- 关键：双向注意力
                )

        # 构建 Mapping
        self_attn_mask_mapping = {
            "full_attention": full_attn_mask,
            "sliding_attention": sliding_attn_mask,
            "encoder_attention_mask": encoder_attention_mask,
        }

        # Create position embeddings to be shared across all decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        all_cross_attentions = () if output_attentions else None

        # Handle early exit for custom layer configurations
        max_needed_layer = float("inf")
        if custom_layers_config is not None and enable_early_exit:
            max_needed_layer = max(custom_layers_config.keys())
            # Force output_attentions to True when early exit is enabled for attention extraction
            output_attentions = True
            if all_cross_attentions is None:
                all_cross_attentions = ()

        # Process through transformer layers
        for index_block, layer_module in enumerate(self.layers):
            layer_outputs = layer_module(
                hidden_states,
                position_embeddings,
                timestep_proj,
                self_attn_mask_mapping[layer_module.attention_type],
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
                encoder_hidden_states,
                self_attn_mask_mapping["encoder_attention_mask"],
                **flash_attn_kwargs,
            )
            hidden_states = layer_outputs[0]

            if output_attentions and self.layers[index_block].use_cross_attention:
                # layer_outputs structure: (hidden_states, self_attn_weights, cross_attn_weights)
                # Extract the last element which is cross_attn_weights
                if len(layer_outputs) >= 3:
                    all_cross_attentions += (layer_outputs[2],)

        if return_hidden_states:
            return hidden_states

        # Extract scale-shift parameters for adaptive output normalization
        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        # Apply adaptive layer norm: norm(x) * (1 + scale) + shift
        hidden_states = (self.norm_out(hidden_states) * (1 + scale) + shift).type_as(
            hidden_states
        )
        # Project output: de-patchify back to original sequence format
        hidden_states = self.proj_out(hidden_states)

        # Crop back to original sequence length to ensure exact length match (remove padding)
        hidden_states = hidden_states[:, :original_seq_len, :]

        outputs = (hidden_states, past_key_values)

        if output_attentions:
            outputs += (all_cross_attentions,)
        return outputs
