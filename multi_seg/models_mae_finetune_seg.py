from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block
# from timm.models.swin_transformer import SwinTransformer3D
from nets.mae.util.pos_embed import get_2d_sincos_pos_embed

import torch.nn as nn
import torch.nn.functional as F

import torch.nn as nn
import torch.nn.functional as F
import math

from utils.help_func import print_var_detail


class DynamicProgressiveSegmentationHead(nn.Module):
    def __init__(self,
                 in_channels,
                 num_classes,
                 input_resolution,  # e.g., 14
                 output_resolution  # e.g., 224
                ):
        super(DynamicProgressiveSegmentationHead, self).__init__()

        assert output_resolution % input_resolution == 0, "Output must be divisible by input resolution!"
        upsample_factor = output_resolution // input_resolution
        num_upsample_layers = int(math.log2(upsample_factor))
        assert 2 ** num_upsample_layers == upsample_factor, "Upsample factor must be power of 2!"

        layers = []
        current_channels = in_channels

        for _ in range(num_upsample_layers - 1):
            layers.append(nn.ConvTranspose2d(current_channels, current_channels // 2, kernel_size=2, stride=2))
            # layers.append(nn.BatchNorm2d(current_channels // 2))
            layers.append(nn.GroupNorm(8, current_channels // 2))
            layers.append(nn.ReLU(inplace=True))
            current_channels = current_channels // 2

        # Last layer: output num_classes channels
        layers.append(nn.ConvTranspose2d(current_channels, num_classes, kernel_size=2, stride=2))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

# class ViTMultiDepthUpscaleDecoder(nn.Module):
#     def __init__(self, in_dims, embed_dim, num_classes, output_resolution=224, num_groups=8):
#         """
#         Args:
#             in_dims: List of input channel dimensions from each depth.
#             embed_dim: Common projection dimension for fusion.
#             num_classes: Number of segmentation classes.
#             output_resolution: Final HxW size to upsample to.
#             num_groups: Number of groups in GroupNorm (default: 8).
#         """
#         super().__init__()
#         self.output_resolution = output_resolution
#
#         self.project_layers = nn.ModuleList([
#             nn.Sequential(
#                 nn.Conv2d(in_dim, embed_dim, kernel_size=1),
#                 nn.GroupNorm(num_groups=min(num_groups, embed_dim), num_channels=embed_dim),
#                 nn.ReLU(inplace=True)
#             ) for in_dim in in_dims
#         ])
#
#         self.fuse = nn.Sequential(
#             nn.Conv2d(embed_dim * len(in_dims), embed_dim, kernel_size=1),
#             nn.GroupNorm(num_groups=min(num_groups, embed_dim), num_channels=embed_dim),
#             nn.ReLU(inplace=True)
#         )
#
#         self.classifier = nn.Conv2d(embed_dim, num_classes, kernel_size=1)
#
#     def forward(self, features):
#         upscaled = []
#         for proj, feat in zip(self.project_layers, features):
#             f = proj(feat)
#             f = F.interpolate(f, size=(self.output_resolution, self.output_resolution),
#                               mode='bilinear', align_corners=False)
#             upscaled.append(f)
#
#         x = torch.cat(upscaled, dim=1)
#         x = self.fuse(x)
#         return self.classifier(x)

import math
import torch
import torch.nn as nn

class ViTMultiDepthUpscaleDecoder(nn.Module):
    def __init__(self, in_dims, base_dim, num_classes, output_resolution=224, num_groups=8):
        super().__init__()
        self.output_resolution = output_resolution
        self.num_classes = num_classes

        def make_upsample_block(in_ch, out_ch):
            return nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
                nn.GroupNorm(num_groups=min(num_groups, out_ch), num_channels=out_ch),
                nn.ReLU(inplace=True)
            )

        self.branches = nn.ModuleList()
        for in_dim in in_dims:
            proj = nn.Conv2d(in_dim, base_dim, kernel_size=1)
            layers = [proj, nn.GroupNorm(num_groups=min(num_groups, base_dim), num_channels=base_dim), nn.ReLU(inplace=True)]

            current_dim = base_dim
            res = 14
            while res < output_resolution:
                next_dim = max(current_dim // 2, 32)  # Avoid going too small
                layers.append(make_upsample_block(current_dim, next_dim))
                current_dim = next_dim
                res *= 2

            self.branches.append(nn.Sequential(*layers))

        self.fuse = nn.Sequential(
            nn.Conv2d(current_dim * len(in_dims), current_dim, kernel_size=1),
            nn.GroupNorm(num_groups=min(num_groups, current_dim), num_channels=current_dim),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Conv2d(current_dim, num_classes, kernel_size=1)

    def forward(self, features):
        # print_var_detail(features, 'features')
        upscaled = [branch(feat) for branch, feat in zip(self.branches, features)]
        # print_var_detail(upscaled)
        x = torch.cat(upscaled, dim=1)
        # print_var_detail(x, 'upscale')
        x = self.fuse(x)
        # print_var_detail(x, 'fuse')
        x = self.classifier(x)
        # print_var_detail(x, 'classifier')
        return x



class MaskedAutoencoderViTMultiSeg(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, num_classes=8, mode='direct',
                 drop=0.1, attn_drop=0.1, drop_path=0.05, feature_blocks = [3, 6, 9, 11]):
        super().__init__()
        self.in_chans = in_chans
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer,drop=drop, attn_drop=attn_drop, drop_path=drop_path)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)  # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

        self.mode = mode  # 'mae' for reconstruction, 'cls' for classification
        self.num_classes = num_classes
        self.reshape_resolution = img_size // patch_size
        if mode == 'CNN':
            self.segmentation_head = DynamicProgressiveSegmentationHead(decoder_embed_dim, num_classes, img_size // patch_size, img_size)
        elif mode == 'direct':
            self.segmentation_head = nn.Linear(decoder_embed_dim, patch_size ** 2 * num_classes, bias=True)
        elif mode == 'ViT-SegFormer':
            self.feature_blocks = feature_blocks
            # self.segmentation_head = ViTMultiDepthUpscaleDecoder(
            #     in_dims=[embed_dim] * len(self.feature_blocks),
            #     embed_dim=768,
            #     num_classes=num_classes,
            #     output_resolution=img_size
            # )
            self.segmentation_head = ViTMultiDepthUpscaleDecoder(
                in_dims=[embed_dim] * len(self.feature_blocks),
                base_dim=768,
                num_classes=num_classes,
                output_resolution=img_size
            )

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    int(self.patch_embed.num_patches ** .5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs, int_chans=3):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], int_chans, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * int_chans))
        return x

    def unpatchify(self, x, out_chans=3):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, out_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], out_chans, h * p, h * p))
        return imgs

    def direct_unpatchify(self, x, patch_size, num_classes):
        """
        For direct prediction: reshape (B, L, patch_size, patch_size, num_classes) --> (B, num_classes, H, W)
        """
        N, L, C = x.shape
        h = w = int(L ** 0.5)
        p = patch_size
        x = x.view(N, h, w, p, p, num_classes)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(N, num_classes, h * p, w * p)
        return x

    def _tokens_to_feature_map(self, x):
        B, N, C = x.shape
        H = W = self.reshape_resolution
        return x.permute(0, 2, 1).contiguous().view(B, C, H, W)

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_encoder_ViT_SegFormer(self, x, mask_ratio = 0.0):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        if self.mode == 'ViT-SegFormer':
            cls_token = self.cls_token + self.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(x.size(0), -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            features = []
            for i, blk in enumerate(self.blocks):
                x = blk(x)
                if i in self.feature_blocks:
                    features.append(self.norm(x[:, 1:, :]))
            return features, None, None
        else:
            raise NotImplementedError("Only ViT-SegFormer mode implemented.")


    def forward_encoder_feature(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        for i, blk in enumerate(self.blocks):
            x = blk(x)
        return x[:, 1:, :]

    def forward_encoder_all_feature(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        features = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            features.append(self.norm(x[:, 1:, :]))
        return features

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_decoder_seg(self, x):
        """
        Decoder for segmentation:
        - Processes CLS + patches through decoder blocks.
        - Drops CLS before reshaping.
        """
        # Embed tokens
        x = self.decoder_embed(x)  # (B, 1+L, decoder_embed_dim)

        # Add positional embedding
        x = x + self.decoder_pos_embed  # (B, 1+L, decoder_embed_dim)

        # Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        if self.mode == 'direct':
            # Remove CLS token AFTER decoding
            x = x[:, 1:, :]  # (B, L, decoder_embed_dim)
            x = self.segmentation_head(x)
            x = self.direct_unpatchify(x, patch_size=self.patch_embed.patch_size[0], num_classes=self.num_classes)
        elif self.mode == 'CNN':
            # Remove CLS token AFTER decoding
            x = x[:, 1:, :]  # (B, L, decoder_embed_dim)
            # Reshape to 2D feature map
            N, L, C = x.shape
            h = w = int(L ** 0.5)
            x = x.permute(0, 2, 1).contiguous().view(N, C, h, w)  # (B, C, H // patch_size, WH // patch_size)
            x = self.segmentation_head(x) # (B, self.num_classes, H, W)
        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs, self.in_chans)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        if self.mode == 'direct' or self.mode == 'CNN':
            latent, _, ids_restore = self.forward_encoder(imgs, mask_ratio=0.0)  # No masking during fine-tuning
            decoder_output = self.forward_decoder_seg(latent)
            return decoder_output # (B, self.num_classes, H, W)
        elif self.mode == 'ViT-SegFormer':
            features, _, _ = self.forward_encoder_ViT_SegFormer(imgs, mask_ratio=0.0)
            features_2d = [self._tokens_to_feature_map(f) for f in features]
            return self.segmentation_head(features_2d)
        else:
            latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
            pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
            loss = self.forward_loss(imgs, pred, mask)
            return loss, pred, mask


def mae_vit_base_patch16_dec512d8b_cls(**kwargs):
    model = MaskedAutoencoderViTMultiSeg(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=8, mode='direct', **kwargs)
    return model

def mae_vit_large_patch16_dec512d8b_cls(**kwargs):
    model = MaskedAutoencoderViTMultiSeg(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=8, mode='direct', **kwargs)
    return model

def mae_vit_huge_patch14_dec512d8b_cls(**kwargs):
    model = MaskedAutoencoderViTMultiSeg(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=8, mode='direct', **kwargs)
    return model

# set recommended archs
mae_vit_base_patch16_cls = mae_vit_base_patch16_dec512d8b_cls  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16_cls = mae_vit_large_patch16_dec512d8b_cls  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14_cls = mae_vit_huge_patch14_dec512d8b_cls  # decoder: 512 dim, 8 blocks

