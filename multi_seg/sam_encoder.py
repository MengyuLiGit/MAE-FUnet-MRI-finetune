import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Any, Dict, List, Tuple

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import resize, to_pil_image  # type: ignore
from typing import Optional, Tuple, Type

from SAM.segment_anything.modeling.common import LayerNorm2d, MLPBlock

# from help_func import #print_var_detail
# from pretrain_mae_rad import mask_ratio


# This class and its supporting functions below lightly adapted from the ViTDet backbone available at: https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py # noqa
class ImageEncoderViTFeature(nn.Module):
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
        export_neck = False,
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
        """
        super().__init__()
        self.img_size = img_size
        self.export_neck = export_neck

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim)
            )

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #print_var_detail(x, 'image encoder x before anything')
        x = self.patch_embed(x)
        #print_var_detail(x, 'image encoder x after patch_embed')
        if self.pos_embed is not None:
            x = x + self.pos_embed
        #print_var_detail(x, 'image encoder x after pos_embed')
        for blk in self.blocks:
            x = blk(x)

        #print_var_detail(x, 'image encoder x before neck')
        if self.export_neck:
            x = self.neck(x.permute(0, 3, 1, 2)) # only output feature map

        return x


class Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        #print_var_detail(x, 'before window_partition x')
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        #print_var_detail(x, 'block attn x')
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))
        #print_var_detail(x, 'window_unpartition x')
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x


class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x


def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(
    windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]
) -> torch.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (
        attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)

    return attn


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x

#
# # ---------- PatchExpand ----------
# from einops import rearrange
# import torch
# import torch.nn as nn
#
# class PatchExpand(nn.Module):
#     def __init__(self, in_dim: int):
#         super().__init__()
#         self.proj = nn.Linear(in_dim, in_dim * 4)  # Upscale by 2x spatially
#         self.norm = nn.LayerNorm(in_dim * 4)
#
#     def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
#         """
#         Args:
#             x: Tensor of shape [B, H*W, C]
#             H, W: current spatial dimensions
#
#         Returns:
#             Tensor of shape [B, H*2, W*2, C]
#         """
#         B, N, C = x.shape
#         assert N == H * W
#         x = self.proj(x)         # [B, N, 4C]
#         x = self.norm(x)         # [B, N, 4C]
#         x = rearrange(x, 'b (h w) (p1 p2 c) -> b (h p1) (w p2) c', h=H, w=W, p1=2, p2=2)
#         return x                 # [B, H*2, W*2, C]
#
#
# # ---------- Cross-Attention ----------
# class CrossAttention(nn.Module):
#     def __init__(self, dim, num_heads=8):
#         super().__init__()
#         self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)
#         self.norm = nn.LayerNorm(dim)
#
#     def forward(self, q_feat, kv_feat):
#         B, H, W, C = q_feat.shape
#         q = q_feat.view(B, H*W, C)
#         kv = kv_feat.view(B, -1, C)
#         out, _ = self.attn(q, kv, kv)
#         out = self.norm(q + out)
#         return out.view(B, H, W, C)
#
# # ---------- DualEncoderSegHead with PatchExpand ----------
# class DualEncoderSegHead(nn.Module):
#     def __init__(self, sam_encoder, mae_encoder, num_classes=7, dim=768, num_heads=8):
#         super().__init__()
#
#         pixel_mean: List[float] = [123.675, 116.28, 103.53]
#         pixel_std: List[float] = [58.395, 57.12, 57.375]
#         self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
#         self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
#
#         self.sam_encoder = sam_encoder
#         self.mae_encoder = mae_encoder
#
#         for p in self.sam_encoder.parameters():
#             p.requires_grad = False
#         for p in self.mae_encoder.parameters():
#             p.requires_grad = False
#
#         # Progressive PatchExpand
#         # Inside DualEncoderSegHead.__init__()
#         self.up1 = PatchExpand(dim)
#         self.up2 = PatchExpand(dim)
#         self.up3 = PatchExpand(dim)
#         self.up4 = PatchExpand(dim)
#         # # Use ConvTranspose2d for large spatial resolution
#         # self.up3 = nn.Sequential(
#         #     nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
#         #     nn.GroupNorm(8, dim),
#         #     nn.ReLU(inplace=True)
#         # )
#         #
#         # self.up4 = nn.Sequential(
#         #     nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
#         #     nn.GroupNorm(8, dim),
#         #     nn.ReLU(inplace=True)
#         # )
#
#         self.cross1 = CrossAttention(dim, num_heads)
#         self.cross2 = CrossAttention(dim, num_heads)
#         self.cross3 = CrossAttention(dim, num_heads)
#         self.cross4 = CrossAttention(dim, num_heads)
#
#         self.head = nn.Sequential(
#             nn.LayerNorm(dim),
#             nn.Linear(dim, num_classes)
#         )
#
#     def forward(self, mri_input):
#         B = mri_input.size(0)
#         # mri_rgb = torch.stack([resize(img, [1024, 1024]) for img in mri_rgb])
#         # Normalize colors
#         mri_rgb = mri_input * 255.0 # assume input is 1-0 float
#
#         # mri_rgb = F.interpolate(mri_rgb, size=(1024, 1024), mode='bilinear', align_corners=False)
#         mri_rgb = resize(mri_rgb, [1024, 1024])
#         mri_rgb = (mri_rgb - self.pixel_mean) / self.pixel_std
#         #print_var_detail(mri_rgb, 'mri_rgb')
#
#
#         # MAE output tokens (assume ViT-SegFormer mode with 14x14 patches)
#         mae_tokens = self.mae_encoder.forward_encoder_ViT_SegFormer(mri_input)[0][-1]  # (B, L, C)
#         # mae_tokens, _, _ = self.mae_encoder.forward_encoder(mri_input, mask_ratio = 0.0)
#         #print_var_detail(mae_tokens, 'mae_tokens')
#         assert mae_tokens.ndim == 3, "Expected [B, L, C] from MAE encoder"
#         H = W = int(mae_tokens.shape[1] ** 0.5)
#
#         # Convert to 2D feature map [B, H, W, C]
#         x = mae_tokens.view(B, H, W, -1)
#
#         # Get SAM features
#         sam = self.sam_encoder(mri_rgb)  # [B, 64, 64, 768]
#         #print_var_detail(sam, 'sam_encoder')
#         # sam = sam.permute(0, 2, 3, 1)    # [B, H, W, C]
#
#         # Stage 1: 14 -> 28
#         #print_var_detail(x, 'x before up1')
#         x = self.up1(x.view(B, -1, x.shape[-1]), H, W)
#         sam1 = F.adaptive_avg_pool2d(sam.permute(0, 3, 1, 2), output_size=(28, 28)).permute(0, 2, 3, 1)
#         #print_var_detail(x, 'x before cross1')
#         #print_var_detail(sam1, 'sam1 before cross1')
#         x = self.cross1(x, sam1)
#         H, W = 28, 28
#
#         # Stage 2: 28 -> 56
#         x = self.up2(x.view(B, -1, x.shape[-1]), H, W)
#         sam2 = F.adaptive_avg_pool2d(sam.permute(0, 3, 1, 2), output_size=(56, 56)).permute(0, 2, 3, 1)
#         x = self.cross2(x, sam2)
#         H, W = 56, 56
#
#         # Stage 3: 56 -> 112
#         x = self.up3(x.permute(0, 3, 1, 2))  # [B, C, H, W]
#         x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
#         sam3 = F.adaptive_avg_pool2d(sam.permute(0, 3, 1, 2), output_size=(112, 112)).permute(0, 2, 3, 1)
#         x = self.cross3(x, sam3)
#         H, W = 112, 112
#
#         # Stage 4: 112 -> 224
#         x = self.up4(x.permute(0, 3, 1, 2))
#         x = x.permute(0, 2, 3, 1)
#         sam4 = F.adaptive_avg_pool2d(sam.permute(0, 3, 1, 2), output_size=(224, 224)).permute(0, 2, 3, 1)
#         x = self.cross4(x, sam4)
#
#         H, W = 224, 224
#         #print_var_detail(x, 'x before head')
#         # Final segmentation head
#         x = self.head(x)  # [B, 224, 224, num_classes]
#         return x.permute(0, 3, 1, 2)  # [B, num_classes, 224, 224]

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchvision.transforms.functional import resize

class PatchExpand(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim * 4)
        self.norm = nn.LayerNorm(out_dim * 4)
        self.out_dim = out_dim

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = self.proj(x)
        x = self.norm(x)
        x = rearrange(x, 'b (h w) (p1 p2 c) -> b (h p1) (w p2) c',
                      h=H, w=W, p1=2, p2=2, c=self.out_dim)
        return x

class CrossAttention(nn.Module):
    def __init__(self, dim_q, dim_kv, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim_q, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim_q)
        self.kv_proj = nn.Linear(dim_kv, dim_q)

    def forward(self, q_feat, kv_feat):
        B, H, W, C = q_feat.shape
        q = q_feat.view(B, H * W, C)
        kv = self.kv_proj(kv_feat.view(B, -1, kv_feat.shape[-1]))
        out, _ = self.attn(q, kv, kv)
        out = self.norm(q + out)
        return out.view(B, H, W, C)

class DualEncoderSegHead(nn.Module):
    def __init__(self, sam_encoder, mae_encoder, num_classes=7, base_dim=768, num_heads=8):
        super().__init__()

        self.sam_encoder = sam_encoder
        self.mae_encoder = mae_encoder

        for p in self.sam_encoder.parameters():
            p.requires_grad = False
        for p in self.mae_encoder.parameters():
            p.requires_grad = False

        self.sam_encoder.eval()
        self.mae_encoder.eval()

        self.pixel_mean = nn.Parameter(torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1), requires_grad=False)
        self.pixel_std = nn.Parameter(torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1), requires_grad=False)

        dims = [base_dim, 384, 192]
        self.up1 = PatchExpand(dims[0], dims[1])  # 14 -> 28
        self.up2 = PatchExpand(dims[1], dims[2])  # 28 -> 56

        self.cross1 = CrossAttention(dims[1], 768, num_heads)
        self.cross2 = CrossAttention(dims[2], 768, num_heads)

        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(dims[2], dims[2] // 2, kernel_size=2, stride=2),  # 56 -> 112
            nn.GroupNorm(8, dims[2] // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(dims[2] // 2, dims[2] // 2, kernel_size=2, stride=2),  # 112 -> 224
            nn.GroupNorm(8, dims[2] // 2),
            nn.ReLU(inplace=True)
        )

        self.head = nn.Sequential(
            nn.LayerNorm(dims[2] // 2),
            nn.Linear(dims[2] // 2, num_classes)
        )

    def forward(self, mri_input):
        if self.mae_encoder.training:
            self.mae_encoder.eval()
        if self.sam_encoder.training:
            self.sam_encoder.eval()

        B = mri_input.size(0)
        mri_rgb = resize(mri_input * 255.0, [1024, 1024])
        mri_rgb = (mri_rgb - self.pixel_mean) / self.pixel_std

        with torch.no_grad():
            mae_tokens = self.mae_encoder.forward_encoder_ViT_SegFormer(mri_input)[0][-1]  # [B, L, C]
            sam = self.sam_encoder(mri_rgb)  # [B, 64, 64, 768]
        H = W = int(mae_tokens.shape[1] ** 0.5)
        x = mae_tokens.view(B, H, W, -1)

        # Stage 1: 14 → 28
        x = self.up1(x.view(B, -1, x.shape[-1]), H, W)
        sam1 = F.adaptive_avg_pool2d(sam.permute(0, 3, 1, 2), (28, 28)).permute(0, 2, 3, 1)
        x = self.cross1(x, sam1)
        H, W = 28, 28

        # Stage 2: 28 → 56
        x = self.up2(x.view(B, -1, x.shape[-1]), H, W)
        sam2 = F.adaptive_avg_pool2d(sam.permute(0, 3, 1, 2), (56, 56)).permute(0, 2, 3, 1)
        x = self.cross2(x, sam2)
        H, W = 56, 56

        # Final upsample 56 → 112 → 224
        x = self.final_up(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = self.head(x)
        return x.permute(0, 3, 1, 2)  # [B, num_classes, 224, 224]


