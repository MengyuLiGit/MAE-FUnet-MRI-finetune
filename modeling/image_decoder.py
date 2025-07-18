# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type

from .common import LayerNorm2d
from help_func import print_var_detail


class ImageDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_params: int=4,
        activation: Type[nn.Module] = nn.GELU,
        output_dim_factor: int = 8,
        SSIM_head_depth: int = 3,
        SSIM_head_hidden_dim: int = 256,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          param_head_depth (int): the depth of the MLP used to predict
            mask quality
          param_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.num_param_tokens = num_params

        # self.num_multimask_outputs = num_multimask_outputs
        #
        self.SSIM_token = nn.Embedding(1, transformer_dim)
        # self.num_mask_tokens = num_multimask_outputs + 1
        # self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // (output_dim_factor // 2), kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // (output_dim_factor // 2)),
            activation(),
            nn.ConvTranspose2d(transformer_dim // (output_dim_factor // 2), transformer_dim // output_dim_factor, kernel_size=2, stride=2),
            activation(),
        )
        # self.output_hypernetworks_mlps = nn.ModuleList(
        #     [
        #         MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
        #         for i in range(self.num_mask_tokens)
        #     ]
        # )
        self.output_hypernetworks_mlp = MLP(transformer_dim * self.num_param_tokens, transformer_dim * self.num_param_tokens, transformer_dim // output_dim_factor, 3)
        #
        self.SSIM_prediction_head = MLP(
            transformer_dim, SSIM_head_hidden_dim, 1, SSIM_head_depth
        ) # return [b, num_params, 1]

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        images = self.predict_images(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
        )

        # Prepare output
        return images #[B, 256/4, H, W]

    def predict_images(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = self.SSIM_token.weight
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        # tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
        # Expand per-image data in batch direction to be per-mask
        # src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        # src = src + dense_prompt_embeddings
        src = image_embeddings
        # pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        pos_src = image_pe
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        # iou_token_out = hs[:, 0, :]
        SSIM_token_out = hs[:, 0, :]
        param_tokens_out = hs[:, 1 : , :]# [b, num_param, transformer_dim]
        # param_tokens_out = hs # [b, num_param, transformer_dim]
        param_tokens_out = param_tokens_out.view(b, 1, -1) # [b, 1, transformer_dim * num_param]



        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        # hyper_in_list: List[torch.Tensor] = []
        # for i in range(self.num_mask_tokens):
        #     hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        # hyper_in = torch.stack(hyper_in_list, dim=1)
        # b, c, h, w = upscaled_embedding.shape
        # masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
        #
        # # Generate mask quality predictions
        SSIM_pred = self.SSIM_prediction_head(SSIM_token_out)
        # b, c, h, w = upscaled_embedding.shape
        # hyper_in = self.output_hypernetworks_mlp(param_tokens_out)
        # image_decoded = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        return upscaled_embedding, SSIM_pred

    def _get_transformer_dim(self):
        return self.transformer_dim


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
