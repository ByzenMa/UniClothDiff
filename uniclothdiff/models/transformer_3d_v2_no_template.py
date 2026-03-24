from typing import Optional

import torch
from torch import nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.embeddings import PatchEmbed, get_1d_sincos_pos_embed_from_grid
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin

from uniclothdiff.models.positional_encoding import ActionEmbedding
from uniclothdiff.registry import MODELS


@MODELS.register_module()
class Transformer3Dv2NoTemplateModel(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        cross_attention_dim: Optional[int] = 1024,
        attention_bias: bool = False,
        sample_size: int = 64,
        patch_size: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = 1000,
        norm_type: str = "ada_norm_zero",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        action_embed_dim: int = 1024,
        num_in_frames: int = 5,
        num_out_frames: int = 1,
        conv_out_kernel: int = 3,
        patchified_input: bool = False,
        point_embed_hidden_dim: int = 512,
    ):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim
        self.patchified_input = patchified_input
        self.num_in_frames = num_in_frames
        self.num_out_frames = num_out_frames

        if not patchified_input:
            self.height = sample_size
            self.width = sample_size

        interpolation_scale = self.config.sample_size // 64
        interpolation_scale = max(interpolation_scale, 1)
        if not patchified_input:
            self.pos_embed = PatchEmbed(
                height=sample_size,
                width=sample_size,
                patch_size=patch_size,
                in_channels=in_channels,
                embed_dim=inner_dim,
                interpolation_scale=interpolation_scale,
            )
            self.point_pos_proj = None
        else:
            self.pos_embed = nn.Sequential(
                nn.Linear(in_channels, point_embed_hidden_dim),
                nn.LayerNorm(point_embed_hidden_dim, elementwise_affine=False, eps=1e-6),
                nn.GELU(),
                nn.Linear(point_embed_hidden_dim, inner_dim),
            )
            self.point_pos_proj = nn.Sequential(
                nn.Linear(3, point_embed_hidden_dim),
                nn.GELU(),
                nn.Linear(point_embed_hidden_dim, inner_dim),
            )

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                    norm_type=norm_type,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                )
                for _ in range(num_layers)
            ]
        )

        self.temporal_transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=None,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                    norm_type=norm_type,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                )
                for _ in range(num_layers)
            ]
        )

        self.out_channels = in_channels if out_channels is None else out_channels
        self.norm_out = nn.LayerNorm(inner_dim, elementwise_affine=False, eps=1e-6)
        if not patchified_input:
            self.proj_out = nn.Linear(inner_dim, patch_size * patch_size * self.out_channels)
        else:
            self.proj_out = nn.Sequential(
                nn.Linear(inner_dim, point_embed_hidden_dim),
                nn.LayerNorm(point_embed_hidden_dim),
                nn.GELU(),
                nn.Linear(point_embed_hidden_dim, self.out_channels),
            )

        self.action_embedding = ActionEmbedding(action_embed_dim)
        temp_pos_embed = get_1d_sincos_pos_embed_from_grid(
            inner_dim, torch.arange(0, num_in_frames).unsqueeze(1)
        )
        self.register_buffer("temp_pos_embed", torch.from_numpy(temp_pos_embed).float().unsqueeze(0), persistent=False)

        if not patchified_input:
            conv_out_padding = (conv_out_kernel - 1) // 2
            self.conv_out = nn.Conv3d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(num_in_frames, conv_out_kernel, conv_out_kernel),
                stride=(num_out_frames, 1, 1),
                padding=(num_out_frames - 1, conv_out_padding, conv_out_padding),
            )
        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        enable_temporal_attentions: bool = True,
        return_dict: bool = True,
    ):
        if not self.patchified_input:
            batch_size, channels, num_frame, height, width = hidden_states.shape
            hidden_states = hidden_states.permute(0, 2, 1, 3, 4).reshape(-1, channels, height, width)
            height, width = (
                hidden_states.shape[-2] // self.config.patch_size,
                hidden_states.shape[-1] // self.config.patch_size,
            )
        else:
            if hidden_states.ndim == 5:
                batch_size, channels, num_frame, num_points, point_dim = hidden_states.shape
                hidden_states = hidden_states.permute(0, 2, 3, 4, 1).reshape(-1, num_points, point_dim)
            elif hidden_states.ndim == 4:
                batch_size, num_frame, num_points, point_dim = hidden_states.shape
                hidden_states = hidden_states.reshape(-1, num_points, point_dim)

        if not self.patchified_input:
            hidden_states = self.pos_embed(hidden_states)
        else:
            point_feature = self.pos_embed(hidden_states)
            point_pos = self.point_pos_proj(hidden_states[..., :3])
            hidden_states = point_feature + point_pos

        num_patches = hidden_states.shape[1]
        encoder_hidden_states = self.action_embedding(encoder_hidden_states)
        encoder_hidden_states_spatial = encoder_hidden_states.repeat_interleave(num_frame, dim=0).view(
            -1, encoder_hidden_states.shape[-2], encoder_hidden_states.shape[-1]
        )

        if len(timestep.shape) < 1:
            timestep = timestep.expand(batch_size)

        timestep_spatial = timestep.repeat_interleave(num_frame, dim=0)
        timestep_temp = timestep.repeat_interleave(num_patches, dim=0)

        for i, (spatial_block, temp_block) in enumerate(
            zip(self.transformer_blocks, self.temporal_transformer_blocks)
        ):
            if self.training and self.gradient_checkpointing:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    spatial_block,
                    hidden_states,
                    None,
                    encoder_hidden_states_spatial,
                    encoder_attention_mask,
                    timestep_spatial,
                    None,
                    None,
                    use_reentrant=False,
                )
            else:
                hidden_states = spatial_block(
                    hidden_states,
                    None,
                    encoder_hidden_states_spatial,
                    encoder_attention_mask,
                    timestep_spatial,
                    None,
                    None,
                )

            if enable_temporal_attentions:
                hidden_states = hidden_states.reshape(
                    batch_size, -1, hidden_states.shape[-2], hidden_states.shape[-1]
                ).permute(0, 2, 1, 3)
                hidden_states = hidden_states.reshape(-1, hidden_states.shape[-2], hidden_states.shape[-1])

                if i == 0 and num_frame > 1:
                    hidden_states = hidden_states + self.temp_pos_embed

                if self.training and self.gradient_checkpointing:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        temp_block,
                        hidden_states,
                        None,
                        None,
                        None,
                        timestep_temp,
                        None,
                        None,
                        use_reentrant=False,
                    )
                else:
                    hidden_states = temp_block(
                        hidden_states,
                        None,
                        None,
                        None,
                        timestep_temp,
                        None,
                        None,
                    )

                hidden_states = hidden_states.reshape(
                    batch_size, -1, hidden_states.shape[-2], hidden_states.shape[-1]
                ).permute(0, 2, 1, 3)
                hidden_states = hidden_states.reshape(-1, hidden_states.shape[-2], hidden_states.shape[-1])

        hidden_states = self.norm_out(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        if not self.patchified_input:
            hidden_states = hidden_states.reshape(
                shape=(-1, height, width, self.config.patch_size, self.config.patch_size, self.out_channels)
            )
            hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
            output = hidden_states.reshape(
                shape=(-1, self.out_channels, height * self.config.patch_size, width * self.config.patch_size)
            )
            output = output.reshape(
                batch_size, -1, output.shape[-3], output.shape[-2], output.shape[-1]
            ).permute(0, 2, 1, 3, 4)
            output = self.conv_out(output).permute(0, 2, 1, 3, 4)
        else:
            hidden_states = hidden_states.reshape(batch_size, -1, *hidden_states.shape[-2:])
            output = hidden_states[:, -self.num_out_frames:, ...]

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
