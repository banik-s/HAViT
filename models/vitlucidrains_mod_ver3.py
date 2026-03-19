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

# ---------------- Attention Block (with inter-layer raw-attention blending) ----------------

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., alpha=0.25):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.alpha = alpha

        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(
                nn.Linear(inner_dim, dim),
                nn.Dropout(dropout)
            )
            if project_out
            else nn.Identity()
        )

    def forward(self, x, h=None):
        """
        x:            (batch, seq_len, dim)
        h (optional): previous raw-attention matrix of shape (batch, heads, seq_len, seq_len)
        """
        x = self.norm(x)

        # Split into query, key, value
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads),
            qkv
        )

        # Raw attention scores
        raw = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (b, heads, n, n)

        # For the first layer, initialize H randomly if not provided
        if h is None:
            h_prev = torch.randn_like(raw)
        else:
            h_prev = h

        # Blend current raw with previous raw-attention
        blended = self.alpha * raw + (1.0 - self.alpha) * h_prev

        # Softmax on blended scores
        attn = torch.softmax(blended, dim=-1)
        attn = self.dropout(attn)

        # Weighted sum of values
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        # Return output plus the new raw-attention matrix for the next layer
        return out, blended  # Return the updated raw-attention (blended) for next layer

    def flops(self, input_shape, layer_index):
        batch_size, seq_len, dim = input_shape
        num_elements = batch_size * seq_len * dim

        # Softmax FLOPS: Approximately 3N - 1
        return 3 * num_elements - 1


# ---------------- Transformer Block ----------------

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout, alpha=0.25),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        """
        x: (batch, seq_len, dim)
        """
        H = None
        for attn, ff in self.layers:
            out, H = attn(x, H)      # get output and new raw-attention
            x = x + out              # residual connection
            x = x + ff(x)            # feed-forward + residual
        return self.norm(x)

    def flops(self, input_shape):
        total_flops = 0
        for layer_idx, (attn, ff) in enumerate(self.layers):
            total_flops += attn.flops(input_shape, layer_idx)
        return total_flops

# ---------------- Vision Transformer ----------------

class vitlucidrains_mod_ver3(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim,
                 pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()

        if isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        else:
            self.image_size = image_size

        if isinstance(patch_size, int):
            self.patch_size = (patch_size, patch_size)
        else:
            self.patch_size = patch_size

        self.pool = pool
        self.dim = dim

        assert pool in {'cls', 'mean'}, 'pool type must be either cls or mean'

        image_h, image_w = pair(self.image_size)
        patch_h, patch_w = pair(self.patch_size)
        assert image_h % patch_h == 0 and image_w % patch_w == 0, \
            'Image dimensions must be divisible by the patch size.'

        num_patches = (image_h // patch_h) * (image_w // patch_w)
        patch_dim = channels * patch_h * patch_w

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_h, p2=patch_w),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token    = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout      = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img)           # (b, num_patches, dim)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)      # add class token
        x = x + self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        return self.mlp_head(x)

    def flops(self, batch_size):
        num_patches = (self.image_size[0] // self.patch_size[0]) * (self.image_size[1] // self.patch_size[1])
        input_shape = (batch_size, num_patches + 1, self.dim)
        total_flops = 0
        for layer_idx, (attn, ff) in enumerate(self.transformer.layers):
            total_flops += attn.flops(input_shape, layer_idx)
        return total_flops

