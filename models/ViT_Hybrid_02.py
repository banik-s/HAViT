import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# Helper function
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# Hybrid Activation Function: ReLU + Polynomial
#def hybrid_activation(x, alpha=0.1, degree=2):
    #"""Applies ReLU in lower layers and Polynomial Activation in higher layers."""
    #return torch.relu(x) + alpha * torch.pow(x, degree)  # Hybrid Activation

# FeedForward Layer
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

    def flops(self, input_shape):
        # No FLOPS calculation here for now, since we focus only on activation
        return 0  # We won't count these in our FLOPS calculation

# Hybrid Attention (Layer-Wise Polynomial + ReLU)
class HybridAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., depth=12):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.depth = depth  # Total number of layers to decide activation type
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if heads > 1 else nn.Identity()

    def forward(self, x, layer_index):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # Apply Hybrid Activation: ReLU in Lower Layers, Polynomial in Higher Layers
        if layer_index < self.depth // 2:
            attn = torch.relu(dots)  # ReLU for early layers
            
        else:
            attn = torch.softmax(dots, dim=-1)  # Softmax activation

        attn = attn / x.shape[1]  # Normalize to prevent overflow

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

    """def flops(self, input_shape, layer_index):
        # Only calculate FLOPS for the activation function
        batch_size, seq_len, dim = input_shape
        activation_flops = batch_size * seq_len * dim  # Hybrid activation FLOPs

        if layer_index < self.depth // 2:
            return activation_flops  # ReLU FLOPS
        else:
            return activation_flops  # softmax FLOPS (same for now)"""
            
    def flops(self, input_shape, layer_index):
        """
        Calculates the FLOPS for ReLU and Softmax activation functions.

        Args:
            input_shape: Tuple representing the input shape (batch_size, seq_len, dim).
            layer_index: Integer representing the layer index.

        Returns:
            Integer representing the FLOPS for the activation function.
        """
        batch_size, seq_len, dim = input_shape
        num_elements = batch_size * seq_len * dim

        if layer_index < self.depth // 2:
            # ReLU FLOPS: Number of comparisons (approx. number of elements)
            return num_elements
        else:
            # Softmax FLOPS: Approximately 3N - 1, where N is num_elements.
            return 3 * num_elements - 1

# Transformer with Hybrid Attention
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])

        for i in range(depth):
            self.layers.append(nn.ModuleList([
                HybridAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout, depth=depth),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        for layer_idx, (attn, ff) in enumerate(self.layers):
            x = attn(x, layer_index=layer_idx) + x
            x = ff(x) + x
        return self.norm(x)

    def flops(self, input_shape):
        total_flops = 0
        for layer_idx, (attn, ff) in enumerate(self.layers):
            total_flops += attn.flops(input_shape, layer_idx)  # Attention Activation FLOPs
            # FeedForward FLOPs are not considered here
        return total_flops

# Vision Transformer with Hybrid Activation (Polynomial + ReLU)
class ViT_Hybrid_02(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim,
                 pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()

        # Ensure image_size is a tuple (height, width)
        if isinstance(image_size, int):  # If image_size is passed as a single integer
            self.image_size = (image_size, image_size)  # Convert to tuple (height, width)
        else:
            self.image_size = image_size  # Otherwise use the passed tuple directly
        
        # Ensure patch_size is a tuple (height, width)
        if isinstance(patch_size, int):  # If patch_size is passed as a single integer
            self.patch_size = (patch_size, patch_size)  # Convert to tuple (height, width)
        else:
            self.patch_size = patch_size  # Otherwise use the passed tuple directly
        
        # Store the pool type ('cls' or 'mean')
        self.pool = pool
        print(f"Image size received: {self.image_size}")  # Debugging line to check the image size
        print(f"Patch size received: {self.patch_size}")  # Debugging line to check the patch size

        self.dim = dim  # Add dim attribute here
        image_height, image_width = pair(self.image_size)
        patch_height, patch_width = pair(self.patch_size)

        # Validate image dimensions against patch size
        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            'Image dimensions must be divisible by the patch size.'

        # Calculate number of patches and patch dimension
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        # Pooling option validation
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # Patch embedding layer
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        # Positional embedding, class token, and dropout
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # Transformer block
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        # Classification head
        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        # Add class token and positional embedding
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        # Pass through transformer layers
        x = self.transformer(x)

        # Pooling and final classification
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        return self.mlp_head(x)

    def flops(self, batch_size):
        # Calculate input shape for ViT
        num_patches = (self.image_size[0] // self.patch_size[0]) * (self.image_size[1] // self.patch_size[1])
        input_shape = (batch_size, num_patches + 1, self.dim)  # Shape: (batch_size, num_patches + 1, dim)
        
        total_flops = 0
        for layer_idx, (attn, ff) in enumerate(self.transformer.layers):
            total_flops += attn.flops(input_shape, layer_idx)  # Attention Activation FLOPs
            # FeedForward FLOPs are not considered here, but can be added if needed
        return total_flops

