import torch
import torch.nn as nn
from torch.nn import functional as F
from nets.mae.util.pos_embed import get_2d_sincos_pos_embed
from help_func import *
class MAEPromptEncoder(nn.Module):
    def __init__(
        self,
        mae_dim: int = 768,
        decoder_dim: int = 256,
        use_learned_pe: bool = True,
        use_mlp: bool = False,
        grid_size: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        self.q_proj = nn.Linear(mae_dim, decoder_dim)
        self.norm = nn.LayerNorm(decoder_dim)
        self.dropout = nn.Dropout(dropout)
        self.use_mlp = use_mlp

        if use_learned_pe:
            self.pos_embed = nn.Parameter(torch.randn(1, grid_size * grid_size, decoder_dim))
        else:
            pe = get_2d_sincos_pos_embed(decoder_dim, grid_size)
            self.register_buffer("pos_embed", torch.tensor(pe).float().unsqueeze(0))

        if use_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(decoder_dim, decoder_dim),
                nn.GELU(),
                nn.Linear(decoder_dim, decoder_dim),
            )

    def forward(self, mae_tokens: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(mae_tokens)
        q = self.norm(q)
        q = self.dropout(q)
        if self.use_mlp:
            q = self.mlp(q)
        return q + self.pos_embed[:, :q.size(1), :]


import torch
from torch import nn
# from SAM.segment_anything.modeling.mask_decoder import MaskDecoder
# from SAM.segment_anything.modeling.two_way_transformer import TwoWayTransformer
from SAM.segment_anything.modeling import MaskDecoder, TwoWayTransformer
from SAM.segment_anything.modeling.common import LayerNorm2d
from SAM.segment_anything.modeling.mask_decoder import MLP
from typing import List, Tuple, Type
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type

from SAM.segment_anything.modeling import TwoWayTransformer
from SAM.segment_anything.modeling.common import LayerNorm2d


class MultiClassMaskDecoder(nn.Module):
    def __init__(
        self,
        transformer_dim: int = 256,
        num_classes: int = 7,
        use_token_type: bool = True,
        transformer: nn.Module = None,
        activation: Type[nn.Module] = nn.GELU,
        output_size: int = 224,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_token_type = use_token_type
        self.output_size = output_size

        self.transformer = transformer or TwoWayTransformer(
            depth=2,
            embedding_dim=transformer_dim,
            num_heads=8,
            mlp_dim=2048,
        )

        self.class_tokens = nn.Embedding(num_classes, transformer_dim)

        if self.use_token_type:
            self.token_type_embed = nn.Embedding(2, transformer_dim)  # 0 = class, 1 = prompt

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 2, kernel_size=2, stride=2),
            nn.GroupNorm(num_groups=min(8, transformer_dim // 2 // 2), num_channels=transformer_dim // 2),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 2, transformer_dim // 4, kernel_size=2, stride=2),
            nn.GroupNorm(num_groups=min(8, transformer_dim // 4 // 2), num_channels=transformer_dim // 4),
            activation(),
        )

        self.smooth = nn.Sequential(
            nn.Conv2d(transformer_dim // 4, transformer_dim // 4, 3, padding=1),
            nn.GroupNorm(num_groups=min(8, transformer_dim // 4 // 2), num_channels=transformer_dim // 4),
            activation(),
        )

        self.final_head = nn.Conv2d(transformer_dim // 4, num_classes, kernel_size=1)

    def forward(
        self,
        image_embeddings: torch.Tensor,        # (B, C, 64, 64)
        image_pe: torch.Tensor,               # (B, C, 64, 64)
        sparse_prompt_embeddings: torch.Tensor,  # (B, N_prompt, C)
        dense_prompt_embeddings: torch.Tensor     # (B, C, 64, 64)
    ) -> torch.Tensor:
        B = sparse_prompt_embeddings.size(0)
        device = sparse_prompt_embeddings.device

        # === Class token setup ===
        class_tokens = self.class_tokens.weight.unsqueeze(0).expand(B, -1, -1)  # (B, num_classes, C)

        # === Token type embedding ===
        if self.use_token_type:
            class_tokens = class_tokens + self.token_type_embed(torch.tensor(0, device=device))
            sparse_prompt_embeddings = sparse_prompt_embeddings + self.token_type_embed(torch.tensor(1, device=device))

        # === Concatenate tokens ===
        tokens = torch.cat([class_tokens, sparse_prompt_embeddings], dim=1)  # (B, num_classes + N_prompt, C)

        # === Transformer decoding ===
        src = image_embeddings + dense_prompt_embeddings  # (B, C, 64, 64)
        hs, src = self.transformer(src, image_pe, tokens)  # hs: (B, N_total, C), src: (B, H*W, C)

        # === Final mask prediction ===
        src = src.transpose(1, 2).view(B, -1, 64, 64)  # (B, C, 64, 64)
        upscaled_feat = self.output_upscaling(src)    # (B, C', 224, 224)
        start = (256 - self.output_size) // 2
        upscaled_feat = upscaled_feat[:, :, start:start + self.output_size, start:start + self.output_size]
        logits = self.smooth(upscaled_feat)
        logits = self.final_head(logits + upscaled_feat)       # (B, num_classes, 224, 224)

        return logits



import torch
import torch.nn as nn
from torchvision.transforms.functional import resize

class MAESAMFusionModel(nn.Module):
    def __init__(self, mae_encoder, sam_encoder, sam_prompt_encoder, img_size=224, num_classes = 14):
        super().__init__()
        self.mae_encoder = mae_encoder
        self.sam_encoder = sam_encoder
        self.prompt_encoder = sam_prompt_encoder
        self.fusion_decoder = MultiClassMaskDecoder(num_classes=num_classes, output_size=img_size)
        self.img_size = img_size

        for m in [self.mae_encoder, self.sam_encoder, self.prompt_encoder]:
            for p in m.parameters():
                p.requires_grad = False
            m.eval()

        self.mae_prompt = MAEPromptEncoder(grid_size = 14)
        self.pixel_mean = nn.Parameter(torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1), requires_grad=False)
        self.pixel_std = nn.Parameter(torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1), requires_grad=False)

    def train(self, mode=True):
        super().train(mode)
        # Force pretrained/frozen modules to stay in eval mode
        self.mae_encoder.eval()
        self.sam_encoder.eval()
        self.prompt_encoder.eval()

    def forward(self, mri_input, multimask_output=False):
        B = mri_input.shape[0]
        mri_rgb = resize(mri_input * 255.0, [1024, 1024])
        mri_rgb = (mri_rgb - self.pixel_mean) / self.pixel_std
        #print_var_detail(mri_rgb, "mri_rgb")

        with torch.no_grad():
            sam_features = self.sam_encoder(mri_rgb)  # (B, 256, 64, 64)
            #print_var_detail(sam_features, "sam_features")
            sam_pe = self.prompt_encoder.get_dense_pe().expand(B, -1, -1, -1)  # (B, 256, 64, 64)
            #print_var_detail(sam_pe, "sam_pe")
            sparse_prompt, dense_prompt = self.prompt_encoder(
                points=None,
                boxes=None,
                masks=None
            )

            mae_tokens = self.mae_encoder.forward_encoder_feature(mri_input)  # (B, L, 768)
            mae_queries = self.mae_prompt(mae_tokens) # (B, L, 256)

        logits = self.fusion_decoder(
            image_embeddings=sam_features,
            image_pe=sam_pe,
            sparse_prompt_embeddings=mae_queries,
            dense_prompt_embeddings=dense_prompt
        )

        # logits = F.interpolate(logits, size=(224, 224), mode="bilinear", align_corners=False)
        return logits  # Shape: [B, num_classes, 224, 224]

