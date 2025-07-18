# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torch import nn

from typing import Any, Optional, Tuple, Type
from help_func import print_var_detail
from .common import LayerNorm2d
from help_func import print_var_detail


class PromptEncoder(nn.Module):
    def __init__(
        self,
        num_head: int,
        num_param_per_head: int,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],
        input_image_size: Tuple[int, int],
        # mask_in_chans: int,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        """
        Encodes prompts for input to SAM's mask decoder.

        Arguments:
          embed_dim (int): The prompts' embedding dimension
          image_embedding_size (tuple(int, int)): The spatial size of the
            image embedding, as (H, W).
          input_image_size (int): The padded size of the image as input
            to the image encoder, as (H, W).
          mask_in_chans (int): The number of hidden channels used for
            encoding input masks.
          activation (nn.Module): The activation to use when encoding
            input masks.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)
        self.num_param_per_head = num_param_per_head

        self.num_head_embeddings = num_head  # pos/neg point + 2 box corners
        head_embeddings = [nn.Embedding(1, embed_dim) for i in range(self.num_head_embeddings)]

        # self.num_param_embeddings = num_head * num_param_per_head  # pos/neg point + 2 box corners
        # param_embeddings = [nn.Embedding(1, embed_dim) for i in range(self.num_param_embeddings)]

        self.head_embeddings = nn.ModuleList(head_embeddings)
        # self.param_embeddings = nn.ModuleList(param_embeddings)
        # self.not_a_param_embed = nn.Embedding(1, embed_dim)
        # self.not_a_point_embed = nn.Embedding(1, embed_dim)
        #
        # self.mask_input_size = (4 * image_embedding_size[0], 4 * image_embedding_size[1])
        # self.mask_downscaling = nn.Sequential(
        #     nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
        #     LayerNorm2d(mask_in_chans // 4),
        #     activation(),
        #     nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
        #     LayerNorm2d(mask_in_chans),
        #     activation(),
        #     nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        # )
        # self.no_mask_embed = nn.Embedding(1, embed_dim)
    # def get_dense_pe(self) -> torch.Tensor:
    #     """
    #     Returns the positional encoding used to encode point prompts,
    #     applied to a dense set of points the shape of the image encoding.
    #
    #     Returns:
    #       torch.Tensor: Positional encoding with shape
    #         1x(embed_dim)x(embedding_h)x(embedding_w)
    #     """
    #     return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def _embed_heads(
        self,
        # params: torch.Tensor, # [B, num_param_per_head]
        labels: torch.Tensor, # [B, ]
    ) -> torch.Tensor:
        """Embeds point prompts."""
        # points = points + 0.5  # Shift to center of pixel
        # if pad:
        #     padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
        #     padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
        #     points = torch.cat([points, padding_point], dim=1)
        #     labels = torch.cat([labels, padding_label], dim=1)
        # param_embedding = torch.zeros((params.shape[0], params.shape[1], self.embed_dim), device=params.device) # [B, num_param_per_head,embed_dim]
        head_embedding = torch.zeros((labels.shape[0], 1, self.embed_dim), device=labels.device) # [B, num_param_per_head,embed_dim]
        for i in range(labels.shape[0]):
            label = labels[i]
            head_embedding[i] = self.head_embeddings[label].weight[0]
            # for j in range(labels.shape[1]):
            #     # print_var_detail(param_embedding[i,j],'param_embedding[i,j]')
            #     # print_var_detail(self.head_embeddings[label].weight)
            #     param_embedding[i,j] += self.head_embeddings[label].weight[0] #add label embedding due to head
            #     param_embedding[i,j] += get_position_encoding_given_key(d =self.embed_dim , k=params[i,j]) #add param value embedding
            #     param_embedding[i,j] += self.param_embeddings[label * self.num_head_embeddings + j].weight[0] #add param class embedding
            #     print("weight[0]",self.param_embeddings[label * self.num_head_embeddings + j].weight[0])
            #     print_var_detail(self.param_embeddings[label * self.num_head_embeddings + j].weight,'weight')

        # replace non_param
        # param_embedding[params == -1] = 0.0
        # param_embedding[params == -1] += self.not_a_param_embed.weight

        # return param_embedding # [B, num_param_per_head,embed_dim]
        return head_embedding # [B, 1, embed_dim]
        #
        #
        #
        #
        # point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)
        # point_embedding[labels == -1] = 0.0
        # point_embedding[labels == -1] += self.not_a_point_embed.weight
        # point_embedding[labels == 0] += self.point_embeddings[0].weight
        # point_embedding[labels == 1] += self.point_embeddings[1].weight
        # return point_embedding
    def forward(
        self,
        params: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Embeds different types of prompts, returning both sparse and dense
        embeddings.

        Arguments:
          points (tuple(torch.Tensor, torch.Tensor) or none): point coordinates
            and labels to embed.
          boxes (torch.Tensor or none): boxes to embed
          masks (torch.Tensor or none): masks to embed

        Returns:
          torch.Tensor: sparse embeddings for the points and boxes, with shape
            BxNx(embed_dim), where N is determined by the number of input points
            and boxes.
          torch.Tensor: dense embeddings for the masks, in the shape
            Bx(embed_dim)x(embed_H)x(embed_W)
        """
        bs = self._get_batch_size(params, labels)
        sparse_embeddings = torch.empty((bs, 0, self.embed_dim), device=self._get_device())
        head_embeddings = self._embed_heads(labels)
        sparse_embeddings = torch.cat([sparse_embeddings, head_embeddings], dim=1)

        return sparse_embeddings

    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def _get_batch_size(
        self,
        params: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
    ) -> int:
        """
        Gets the batch size of the output given the batch size of the input prompts.
        """
        if params is not None:
            return params.shape[0]
        elif labels is not None:
            return labels.shape[0]
        else:
            return 1

    def _get_device(self) -> torch.device:
        return self.head_embeddings[0].weight.device


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C

def get_position_encoding_given_key(d, k, n=torch.tensor([10000.0])):
    P = torch.zeros((d))
    for i in range(int(d/2)):
        denominator = torch.pow(n, 2*i/d)
        P[2*i] = torch.sin((k)/denominator)
        P[2*i+1] = torch.cos((k)/denominator)
    return P

# P = get_position_encoding_given_key(d=6, k = 3)
# print(P)
# print_var_detail(P)