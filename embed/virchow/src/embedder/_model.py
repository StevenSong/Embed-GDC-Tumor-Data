"""Virchow tile encoder.

Normalization and Virchow's embedding step (concat of the class token with the mean of
the patch tokens) are baked into the module, so it takes uint8 tiles and returns
(B, 2560) float32 embeddings. Kept in eager mode; the caller runs it under autocast.

Source: https://huggingface.co/paige-ai/Virchow
"""

import timm
import torch
import torch.nn as nn
from timm.layers.mlp import SwiGLUPacked

HF_HUB_ID = "hf-hub:paige-ai/Virchow"
TILE_PX = 224
TILE_UM = 112  # Virchow works at 20x i.e. 0.5 MPP
TARGET_MPP = TILE_UM / TILE_PX
EMBED_DIM = 2560
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class WrappedVirchow(nn.Module):
    """
    Expects 224x224 px (20x @ 0.5MPP) RGB input images as uint8 Tensor of shape (B, C, H, W),
    where B is batch size, C = 3 channels (ordered RGB), and H = W = 224.
    """

    def __init__(self):
        super().__init__()
        self.virchow = timm.create_model(
            HF_HUB_ID,
            pretrained=True,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )

        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x.float() / 255.0 - self.mean) / self.std  # (B, 3, 224, 224)
        x = self.virchow(x)  # (B, 257, 1280)

        cls_emb = x[:, 0]  # (B, 1280)
        patch_embs = x[:, 1:]  # (B, 256, 1280)

        # concatenate class token and average pool of patch tokens
        x = torch.cat([cls_emb, patch_embs.mean(1)], dim=-1)  # (B, 2560)

        return x
