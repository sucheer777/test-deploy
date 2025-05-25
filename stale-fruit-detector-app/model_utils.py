import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import logging
import os
import numpy as np
import math

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_utils.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """Truncated normal initialization"""
    # Get values from a normal distribution
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < b) & (tmp > a)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor

# Add trunc_normal_ to nn.init if it doesn't exist
if not hasattr(nn.init, 'trunc_normal_'):
    nn.init.trunc_normal_ = trunc_normal_

# Shelf life class names with more detailed conditions
shelf_life_class_names = [
    'apple_fresh', 'apple_ripe', 'apple_overripe', 'apple_stale',
    'banana_fresh', 'banana_ripe', 'banana_overripe', 'banana_stale',
    'orange_fresh', 'orange_ripe', 'orange_overripe', 'orange_stale'
]

# Detailed shelf life data with conditions and storage recommendations
shelf_life_data = {
    'apple': {
        'fresh': '2-3 weeks in refrigerator (crisp, firm, bright color)',
        'ripe': '1-2 weeks in refrigerator (slightly softer but still firm)',
        'overripe': '3-5 days (softer texture, still edible)',
        'stale': 'Not recommended for consumption'
    },
    'banana': {
        'fresh': '5-7 days at room temperature (green to yellow)',
        'ripe': '2-3 days at room temperature (yellow with brown spots)',
        'overripe': '1-2 days (brown spots, very soft, good for baking)',
        'stale': 'Not recommended for consumption'
    },
    'orange': {
        'fresh': '2-3 weeks in refrigerator (firm, bright color)',
        'ripe': '1-2 weeks in refrigerator (slightly softer)',
        'overripe': '3-5 days (soft spots starting to appear)',
        'stale': 'Not recommended for consumption'
    }
}

# Visual indicators for each condition
condition_indicators = {
    'fresh': ['bright color', 'firm texture', 'no blemishes', 'natural shine'],
    'ripe': ['full color', 'slight give when pressed', 'aromatic', 'peak flavor'],
    'overripe': ['darker color', 'soft spots', 'strong aroma', 'very soft'],
    'stale': ['discoloration', 'mold', 'mushy texture', 'off odor']
}

# Keywords for fruit detection
fruit_keywords = [
    'apple', 'banana', 'orange', 'fruit',
    'granny smith', 'macintosh', 'golden delicious', 'fuji', 'honeycrisp',  # apple varieties
    'cavendish', 'plantain', 'lady finger',  # banana varieties
    'citrus', 'mandarin', 'tangerine', 'clementine', 'valencia',  # orange/citrus varieties
    'produce', 'food', 'edible', 'fresh',  # general food terms
    'ripe', 'unripe', 'raw',  # ripeness indicators
    'peel', 'skin', 'flesh',  # fruit parts
    'tropical', 'organic'  # descriptive terms
]

def get_condition_from_confidence(confidence_scores):
    """
    Determine fruit condition based on confidence scores.
    confidence_scores should be a dict with scores for each condition.
    """
    # Sort conditions by confidence score
    sorted_conditions = sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True)
    top_condition = sorted_conditions[0][0]
    top_confidence = sorted_conditions[0][1]
    
    # Log the decision process
    logger.info(f"Condition confidence scores: {confidence_scores}")
    logger.info(f"Selected condition: {top_condition} with confidence: {top_confidence:.2f}%")
    
    return top_condition, top_confidence

# Configuration for ViT
PATCH_SIZE = 16
IMAGE_WIDTH = 224
IMAGE_HEIGHT = IMAGE_WIDTH
IMAGE_CHANNELS = 3
EMBEDDING_DIMS = 768  # Standard ViT base config
NUM_OF_PATCHES = (IMAGE_WIDTH * IMAGE_HEIGHT) // (PATCH_SIZE ** 2)

# Patch Embedding Layer for ViT
class PatchEmbeddingLayer(nn.Module):
    def __init__(self, in_channels, patch_size, embedding_dim):
        super().__init__()
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=embedding_dim,
                                    kernel_size=patch_size, stride=patch_size)

        self.class_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.position_embeddings = nn.Parameter(torch.randn(1, NUM_OF_PATCHES + 1, embedding_dim))

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.conv_layer(x)  # (B, E, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, N, E)
        class_token = self.class_token.expand(batch_size, -1, -1)  # (B, 1, E)
        x = torch.cat((class_token, x), dim=1)  # (B, N+1, E)
        x = x + self.position_embeddings  # add positional encoding
        return x

# Multi-Head Self Attention Block for ViT
class MultiHeadSelfAttentionBlock(nn.Module):
    def __init__(self, embedding_dims=768, num_heads=12, attn_dropout=0.0):
        super().__init__()
        self.layernorm = nn.LayerNorm(embedding_dims)
        self.multiheadattention = nn.MultiheadAttention(embed_dim=embedding_dims,
                                                        num_heads=num_heads,
                                                        dropout=attn_dropout,
                                                        batch_first=True)

    def forward(self, x):
        norm_x = self.layernorm(x)
        attn_output, _ = self.multiheadattention(norm_x, norm_x, norm_x)
        return x + attn_output  # residual connection

# MLP Block for ViT
class MachineLearningPerceptronBlock(nn.Module):
    def __init__(self, embedding_dims, mlp_size, mlp_dropout):
        super().__init__()
        self.layernorm = nn.LayerNorm(embedding_dims)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dims, mlp_size),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_size, embedding_dims),
            nn.Dropout(mlp_dropout)
        )

    def forward(self, x):
        return x + self.mlp(self.layernorm(x))  # residual connection

# Transformer Block for ViT
class TransformerBlock(nn.Module):
    def __init__(self, embedding_dims=768, mlp_dropout=0.1, attn_dropout=0.0, mlp_size=3072, num_heads=12):
        super().__init__()
        self.msa_block = MultiHeadSelfAttentionBlock(embedding_dims, num_heads, attn_dropout)
        self.mlp_block = MachineLearningPerceptronBlock(embedding_dims, mlp_size, mlp_dropout)

    def forward(self, x):
        x = self.msa_block(x)
        x = self.mlp_block(x)
        return x

# Vision Transformer
class ViT(nn.Module):
    def __init__(self, img_size=224, in_channels=3, patch_size=16, embedding_dims=768,
                 num_transformer_layers=12, mlp_dropout=0.1, attn_dropout=0.0,
                 mlp_size=3072, num_heads=12, num_classes=1000):
        super().__init__()

        self.patch_embedding_layer = PatchEmbeddingLayer(in_channels=in_channels,
                                                         patch_size=patch_size,
                                                         embedding_dim=embedding_dims)

        self.transformer_encoder = nn.Sequential(*[
            TransformerBlock(embedding_dims, mlp_dropout, attn_dropout, mlp_size, num_heads)
            for _ in range(num_transformer_layers)
        ])

        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dims),
            nn.Linear(embedding_dims, num_classes)
        )

    def forward(self, x):
        x = self.patch_embedding_layer(x)
        x = self.transformer_encoder(x)
        x = x[:, 0]  # take class token
        x = self.classifier(x)
        return x

# Window Partition and Reverse Functions for Swin
def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

# MLP Module for Swin
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# Window Attention Module for Swin
class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        coords = torch.stack(torch.meshgrid(
            torch.arange(window_size), torch.arange(window_size), indexing='ij'
        ))
        coords_flatten = coords.flatten(1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1
        )
        attn += relative_position_bias.permute(2, 0, 1).unsqueeze(0)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# Swin Transformer Block
class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0, mlp_ratio=4.0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, window_size=window_size, num_heads=num_heads
        )

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        attn_windows = self.attn(x_windows)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)
        x = x + self.norm1(x)  # Residual connection for attention
        x = x + self.mlp(self.norm2(x))  # Residual connection for MLP
        return x

# Patch Embedding for Swin
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.patches_resolution = [img_size // patch_size, img_size // patch_size]
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

# Patch Merging for Swin
class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim):
        super().__init__()
        self.input_resolution = input_resolution
        self.reduction = nn.Linear(4 * dim, 2 * dim)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        
        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)

        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x

# Basic Layer for Swin
class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(depth):
            shift_size = 0 if i % 2 == 0 else window_size // 2
            block = SwinTransformerBlock(
                dim, input_resolution, num_heads, window_size, shift_size
            )
            self.blocks.append(block)
        self.downsample = PatchMerging(input_resolution, dim)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        x = self.downsample(x)
        return x

# Swin Transformer
class SwinTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96,
                 depths=[2, 2], num_heads=[3, 6], window_size=7, num_classes=2):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.layers = nn.ModuleList()

        resolution = img_size // patch_size
        for i_layer in range(len(depths)):
            layer = BasicLayer(
                dim=embed_dim * (2 ** i_layer),
                input_resolution=(resolution // (2 ** i_layer), resolution // (2 ** i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size
            )
            self.layers.append(layer)

        # Corrected LayerNorm dimension
        self.norm = nn.LayerNorm(embed_dim * (2 ** len(depths)))
        self.head = nn.Linear(embed_dim * (2 ** len(depths)), num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.head(x)
        return x
