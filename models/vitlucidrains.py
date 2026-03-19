import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# ---------------- Helper Functions ----------------

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# ---------------- FeedForward Block ----------------

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

# ---------------- Attention Block ----------------

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        # Split into query, key, value
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        # Compute scaled dot-product attention
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # Use Softmax as the activation function in attention
        attn = torch.softmax(dots, dim=-1)  # Softmax activation

        # Apply dropout and compute the weighted sum of values
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

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

        
        # Softmax FLOPS: Approximately 3N - 1, where N is num_elements.
        return 3 * num_elements - 1


# ---------------- Transformer Block ----------------

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        b, n, _ = x.shape  # Get batch size and sequence length from input
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

    def flops(self, input_shape):
        total_flops = 0
        for attn, ff in self.layers:
            total_flops += attn.flops(input_shape)  # Attention FLOPs (activation FLOPS only)
        return total_flops

# ---------------- Vision Transformer ----------------

class vitlucidrains(nn.Module):
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

