import torch
import torch.nn as nn
import torch.nn.functional as F


class MAEFeatureProjector(nn.Module):
    # Includes InstanceNorm2d before reshaping for stability
    """
    Projects MAE embeddings from shape [B, 196, 768] to feature maps with shape [B, out_dim, H, W].
    """
    def __init__(self, in_dim=768, out_dim=64, target_size=(64, 64)):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm = nn.InstanceNorm2d(out_dim, affine=True)
        self.target_size = target_size

    def forward(self, x):  # [B, 196, 768]
        B, N, C = x.shape
        H = W = int(N ** 0.5)  # typically 14 for ViT-B/16
        x = self.proj(x)  # [B, 196, out_dim]
        x = x.permute(0, 2, 1).reshape(B, -1, H, W)
        x = self.norm(x)  # [B, out_dim, 14, 14]
        x = F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False)
        return x  # [B, out_dim, H_tgt, W_tgt]


class FusionBlock(nn.Module):
    """
    Fuses U-Net feature maps with MAE feature maps using specified strategy.
    Supported modes: 'add', 'concat', 'attention'

    For 'attention' mode, if skip_connect=True, you can specify skip_type='add' or 'concat'.
    """
    def __init__(self, mode='add', in_ch=None, fuse_ch=None, skip_connect=False, skip_type='add'):
        super().__init__()
        self.mode = mode
        self.skip_connect = skip_connect
        self.skip_type = skip_type

        if mode == 'concat':
            self.conv = nn.Conv2d(in_ch + fuse_ch, in_ch, kernel_size=1)

        elif mode == 'attention':
            self.attn = nn.MultiheadAttention(embed_dim=in_ch, num_heads=4, batch_first=True)
            if skip_connect and skip_type == 'concat':
                self.skip_conv = nn.Conv2d(in_ch * 2, in_ch, kernel_size=1)

    def forward(self, x, fuse):
        if self.mode == 'add':
            return x + fuse

        elif self.mode == 'concat':
            out = torch.cat([x, fuse], dim=1)
            return self.conv(out)

        elif self.mode == 'attention':
            B, C, H, W = x.shape
            x_flat = x.flatten(2).transpose(1, 2)       # [B, H*W, C]
            fuse_flat = fuse.flatten(2).transpose(1, 2) # [B, H*W, C]
            attn_out, _ = self.attn(x_flat, fuse_flat, fuse_flat)
            attn_out = attn_out.transpose(1, 2).reshape(B, C, H, W)

            if self.skip_connect:
                if self.skip_type == 'add':
                    return x + attn_out
                elif self.skip_type == 'concat':
                    concat = torch.cat([x, attn_out], dim=1)
                    return self.skip_conv(concat)
                else:
                    raise ValueError(f"Unsupported skip_type: {self.skip_type}")
            else:
                return attn_out

        else:
            raise ValueError(f"Unsupported fusion mode: {self.mode}")



class UnetWithMAEFusion(nn.Module):
    """
    U-Net with integrated MAE encoder. Accepts a pretrained MAE model to extract intermediate features.
    """
    """
    Wraps a standard U-Net and fuses MAE transformer block outputs into each U-Net stage.
    Uses specific MAE layer indices for each encoder-decoder stage and bottleneck.
    fusion_mode: 'add', 'concat', or 'attention'
    """
    def __init__(self, unet, mae_model, custom_fusion_modes=['add', 'add', 'add', 'attention', 'attention', 'attention', 'add', 'add', 'add'],
                 mae_indices = [0, 2, 5, 8, 11, 8, 5, 2, 0] , skip_connect = True, skip_type = 'add'):
        super().__init__()
        self.unet = unet
        self.mae = mae_model
        self.custom_fusion_modes = custom_fusion_modes

        ch = unet.chans
        self.mae_indices = mae_indices  # encoder: 0,2,5,8; bottleneck:11; decoder: 8,5,2,0

        self.stage_channels = [
            ch * (2 ** i) for i in range(unet.num_pool_layers)  # down: 32, 64, 128, 256
        ] + [
            ch * (2 ** unet.num_pool_layers)  # bottleneck: 512
        ] + [
            ch * (2 ** (unet.num_pool_layers - 1 - i)) for i in range(unet.num_pool_layers)  # up: 256, 128, 64, 32
        ]

        self.proj_layers = nn.ModuleList([
            MAEFeatureProjector(768, c, target_size=(224 // (2 ** i), 224 // (2 ** i)))
            for i, c in enumerate(self.stage_channels[:unet.num_pool_layers])
        ] + [
            MAEFeatureProjector(768, self.stage_channels[unet.num_pool_layers], target_size=(14, 14))
        ] + [
            MAEFeatureProjector(768, c, target_size=(224 // (2 ** (unet.num_pool_layers - 1 - i)),
                                                    224 // (2 ** (unet.num_pool_layers - 1 - i))))
            for i, c in enumerate(self.stage_channels[unet.num_pool_layers + 1:])
        ])

        self.fusion_blocks = nn.ModuleList([
            FusionBlock(self.custom_fusion_modes[i], c, c, skip_connect=skip_connect, skip_type = skip_type) for i, c in enumerate(self.stage_channels)
        ])
        self.if_freeze_mae = not any(p.requires_grad for p in self.mae.parameters())
        if self.if_freeze_mae:
            self.mae.eval()

    def forward(self, x):
        # Extract all 12 MAE intermediate features
        # mae_feats = self.mae.forward_encoder_all_feature(x)
        if self.if_freeze_mae:
            self.mae.eval()
            with torch.no_grad():
                mae_feats = self.mae.forward_encoder_all_feature(x)
        else:
            mae_feats = self.mae.forward_encoder_all_feature(x)
        # mae_feats: list of 12 outputs from each transformer block
        assert len(mae_feats) == 12

        fused_feats = [proj(mae_feats[idx]) for proj, idx in zip(self.proj_layers, self.mae_indices)]
        stack = []
        output = x

        # === Downsampling path ===
        for i, layer in enumerate(self.unet.down_sample_layers):
            output = layer(output)
            output = self.fusion_blocks[i](output, fused_feats[i])
            if self.unet.use_attention:
                output = self.unet.down_att_layers[i](output)
            stack.append(output)
            output = F.avg_pool2d(output, 2)

        # === Bottleneck ===
        output = self.unet.conv(output)
        output = self.fusion_blocks[self.unet.num_pool_layers](output, fused_feats[self.unet.num_pool_layers])
        if self.unet.use_attention:
            output = self.unet.conv_att(output)

        # === Upsampling path ===
        for i in range(self.unet.num_pool_layers):
            downsample_layer = stack.pop()
            output = self.unet.up_transpose_conv[i](output)

            # Padding for odd-sized inputs
            padding = [0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1
            if sum(padding) > 0:
                output = F.pad(output, padding, "reflect")

            output = torch.cat([output, downsample_layer], dim=1)
            output = self.unet.up_conv[i](output)
            output = self.fusion_blocks[self.unet.num_pool_layers + 1 + i](output, fused_feats[self.unet.num_pool_layers + 1 + i])
            if self.unet.use_attention:
                output = self.unet.up_att[i](output)

        output = self.unet.out_conv(output)
        if self.unet.if_classify:
            output = torch.flatten(output, 1)
            output = self.unet.out_classifier(output)

        return output
