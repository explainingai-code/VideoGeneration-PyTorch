import torch
import torch.nn as nn
from model.patch_embed import PatchEmbedding
from model.transformer_layer import TransformerLayer
from einops import rearrange, repeat


def get_time_embedding(time_steps, temb_dim):
    r"""
    Convert time steps tensor into an embedding using the
    sinusoidal time embedding formula
    :param time_steps: 1D tensor of length batch size
    :param temb_dim: Dimension of the embedding
    :return: BxD embedding representation of B time steps
    """
    assert temb_dim % 2 == 0, "time embedding dimension must be divisible by 2"

    # factor = 10000^(2i/d_model)
    factor = 10000 ** ((torch.arange(
        start=0,
        end=temb_dim // 2,
        dtype=torch.float32,
        device=time_steps.device) / (temb_dim // 2))
    )

    # pos / factor
    # timesteps B -> B, 1 -> B, temb_dim
    t_emb = time_steps[:, None].repeat(1, temb_dim // 2) / factor
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
    return t_emb


class DITVideo(nn.Module):
    r"""
    Class for the Latte Model, which makes use of alternate spatial
    and temporal encoder layers, each of which are DiT blocks.
    """
    def __init__(self, frame_height, frame_width, im_channels, num_frames, config):
        super().__init__()

        num_layers = config['num_layers']
        self.image_height = frame_height
        self.image_width = frame_width
        self.im_channels = im_channels
        self.hidden_size = config['hidden_size']
        self.patch_height = config['patch_size']
        self.patch_width = config['patch_size']
        self.num_frames = num_frames

        self.timestep_emb_dim = config['timestep_emb_dim']

        # Number of patches along height and width
        self.nh = self.image_height // self.patch_height
        self.nw = self.image_width // self.patch_width

        # Patch Embedding Block
        self.patch_embed_layer = PatchEmbedding(image_height=self.image_height,
                                                image_width=self.image_width,
                                                im_channels=self.im_channels,
                                                patch_height=self.patch_height,
                                                patch_width=self.patch_width,
                                                hidden_size=self.hidden_size)

        # Initial projection from sinusoidal time embedding
        self.t_proj = nn.Sequential(
            nn.Linear(self.timestep_emb_dim, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )

        # All Transformer Layers
        self.layers = nn.ModuleList([
            TransformerLayer(config) for _ in range(num_layers)
        ])

        # Final normalization for unpatchify block
        self.norm = nn.LayerNorm(self.hidden_size, elementwise_affine=False, eps=1E-6)

        # Scale and Shift parameters for the norm
        self.adaptive_norm_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_size, 2 * self.hidden_size, bias=True)
        )

        # Final Linear Layer
        self.proj_out = nn.Linear(self.hidden_size,
                                  self.patch_height * self.patch_width * self.im_channels)

        ############################
        # DiT Layer Initialization #
        ############################
        nn.init.normal_(self.t_proj[0].weight, std=0.02)
        nn.init.normal_(self.t_proj[2].weight, std=0.02)

        nn.init.constant_(self.adaptive_norm_layer[-1].weight, 0)
        nn.init.constant_(self.adaptive_norm_layer[-1].bias, 0)

        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

    def forward(self, x, t, num_images=0):
        r"""
        Forward method of our ditv model which predicts the noise
        :param x: input noisy image
        :param t: timestep of noise
        :param num_images: if joint training then this is number
        of images appended to video frames
        :return:
        """
        # Shape of x is Batch_size x (num_frames + num_images) x Channels x H x W
        B, F, C, H, W = x.shape

        ##################
        # Patchify Block #
        ##################
        # rearrange to (Batch_size * (num_frames + num_images)) x Channels x H x W
        x = rearrange(x, 'b f c h w -> (b f) c h w')
        out = self.patch_embed_layer(x)

        # out->(Batch_size * (num_frames + num_images)) x num_patch_tokens x hidden_size
        num_patch_tokens = out.shape[1]

        # Compute Timestep representation
        # t_emb -> (Batch, timestep_emb_dim)
        t_emb = get_time_embedding(torch.as_tensor(t).long(), self.timestep_emb_dim)
        # (Batch, timestep_emb_dim) -> (Batch, hidden_size)
        t_emb = self.t_proj(t_emb)

        # Timestep embedding will be Batch_size x hidden_size
        # We repeat it to get different timestep shapes for spatial and temporal layers
        # For spatial -> (Batch size * (num_frames + num_images)) x hidden_size
        # For temporal -> (Batch size * num_patch_tokens) x hidden_size
        t_emb_spatial = repeat(t_emb, 'b d -> (b f) d',
                               f=self.num_frames+num_images)
        t_emb_temporal = repeat(t_emb, 'b d -> (b p) d', p=num_patch_tokens)

        # get temporal embedding from 0-num_frames(16)
        frame_pos = torch.arange(self.num_frames, dtype=torch.float32, device=x.device)
        frame_emb = get_time_embedding(frame_pos, self.hidden_size)
        # frame_emb -> (16 x hidden_size)

        # Loop over all transformer layers
        for layer_idx in range(0, len(self.layers), 2):
            spatial_layer = self.layers[layer_idx]
            temporal_layer = self.layers[layer_idx+1]

            # out->(Batch_size * (num_frames+num_images)) x num_patch_tokens x hidden_size

            #################
            # Spatial Layer #
            #################
            # position embedding is already added in patch embedding layer
            out = spatial_layer(out, t_emb_spatial)

            ##################
            # Temporal Layer #
            ##################
            # rearrange to (B * patch_tokens) x (num_frames+num_images) x hidden_size
            out = rearrange(out, '(b f) p d -> (b p) f d', b=B)

            # Separate the video tokens and image tokens
            out_video = out[:, :self.num_frames, :]
            out_images = out[:, self.num_frames:, :]

            # Add temporal embedding to video tokens
            # but only if first temporal layer
            if layer_idx == 0:
                out_video = out_video + frame_emb
            # Call temporal layer
            out_video = temporal_layer(out_video, t_emb_temporal)

            # Concatenate the image tokens back to the new video output
            out = torch.cat([out_video, out_images], dim=1)

            # Rearrange to (B * (num_frames+num_images)) x num_patch_tokens x hidden_size
            out = rearrange(out, '(b p) f d -> (b f) p d',
                            f=self.num_frames+num_images, p=num_patch_tokens)

        # Shift and scale predictions for output normalization
        pre_mlp_shift, pre_mlp_scale = (self.adaptive_norm_layer(t_emb_spatial).
                                        chunk(2, dim=1))
        out = (self.norm(out) * (1 + pre_mlp_scale.unsqueeze(1)) +
               pre_mlp_shift.unsqueeze(1))

        # Unpatchify
        # Batch_size * (num_frames+num_images)) x patches x hidden_size
        # -> (B * (num_frames+num_images)) x patches x (patch height*patch width*channels)
        out = self.proj_out(out)
        out = rearrange(out, 'b (nh nw) (ph pw c) -> b c (nh ph) (nw pw)',
                        ph=self.patch_height,
                        pw=self.patch_width,
                        nw=self.nw,
                        nh=self.nh)
        # out -> (Batch_size * (num_frames+num_images)) x channels x h x w
        out = out.reshape((B, F, C, H, W))
        return out
