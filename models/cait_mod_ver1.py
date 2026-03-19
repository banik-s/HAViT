import torch
from torch import nn, einsum
import torch.nn.functional as F
from random import randrange

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# ---------------- Helpers ----------------

def exists(val):
    return val is not None

def dropout_layers(layers, dropout):
    if dropout == 0:
        return layers
    num_layers = len(layers)
    to_drop = torch.zeros(num_layers).uniform_(0., 1.) < dropout
    # ensure at least one layer remains
    if all(to_drop):
        to_drop[randrange(num_layers)] = False
    return [layer for (layer, drop) in zip(layers, to_drop) if not drop]

# ---------------- LayerScale ----------------

class LayerScale(nn.Module):
    def __init__(self, dim, fn, depth):
        super().__init__()
        if depth <= 18:
            init_eps = 0.1
        elif depth <= 24:
            init_eps = 1e-5
        else:
            init_eps = 1e-6
        scale = torch.zeros(1, 1, dim).fill_(init_eps)
        self.scale = nn.Parameter(scale)
        self.fn = fn
    
    def forward(self, x, **kwargs):
        result = self.fn(x, **kwargs)
        
        # Handle tuple return (for attention layers)
        if isinstance(result, tuple):
            output, *rest = result
            scaled_output = output * self.scale
            return (scaled_output, *rest)
        
        # Handle single tensor return (for feedforward layers)
        return result * self.scale


# ---------------- FeedForward ----------------

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
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

# ---------------- Blended Attention ----------------

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., alpha = 0.45):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.alpha = alpha

        self.norm = nn.LayerNorm(dim)
        self.to_q   = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv  = nn.Linear(dim, inner_dim * 2, bias=False)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        # talking‐heads parameters
        self.mix_heads_pre_attn  = nn.Parameter(torch.randn(heads, heads))
        self.mix_heads_post_attn = nn.Parameter(torch.randn(heads, heads))

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context = None, h = None):
        """
        x:       (batch, seq_len, dim)
        context: (batch, seq_len_ctx, dim) or None
        h:       previous raw-attention (batch, heads, seq_len, seq_len)
        """
        b, n, _ = x.shape

        # 1) normalize and build q, k, v
        x_norm = self.norm(x)
        ctx    = x_norm if context is None else torch.cat((x_norm, context), dim = 1)

        q = self.to_q(x_norm)
        k, v = self.to_kv(ctx).chunk(2, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b m (h d) -> b h m d', h = self.heads), (q, k, v))

        # 2) raw attention and initialize h_prev
        raw = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        if h is None:
            h_prev = torch.randn_like(raw)
        else:
            h_prev = h

        # 3) blend and talking‐heads (pre-softmax)
        blended = self.alpha * raw + (1 - self.alpha) * h_prev
        dots    = einsum('b h i j, h g -> b g i j', blended, self.mix_heads_pre_attn)

        # 4) softmax + dropout
        attn = self.attend(dots)
        attn = self.dropout(attn)

        # 5) talking‐heads (post-softmax)
        attn = einsum('b h i j, h g -> b g i j', attn, self.mix_heads_post_attn)

        # 6) attend and project out
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        # return both the output and the new raw-attention
        return out, blended

# ---------------- Blended Transformer ----------------

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim,
                 dropout = 0., layer_dropout = 0.):
        super().__init__()
        self.layer_dropout = layer_dropout
        self.layers = nn.ModuleList([])

        for ind in range(depth):
            attn = Attention(dim, heads = heads, dim_head = dim_head,
                             dropout = dropout, alpha = 0.45)
            ff   = FeedForward(dim, mlp_dim, dropout = dropout)
            # wrap with LayerScale if you want:
            self.layers.append(nn.ModuleList([
                LayerScale(dim, attn, depth = ind+1),
                LayerScale(dim, ff,   depth = ind+1)
            ]))

    def forward(self, x, context = None):
        """
        x:       (b, seq_len, dim)
        context: optional context for kv in cls_transformer
        """
        H = None
        layers = dropout_layers(self.layers, dropout = self.layer_dropout)

        for attn, ff in layers:
            out, H = attn(x, context = context, h = H)
            x = x + out
            x = x + ff(x)

        return x

# ---------------- CaiT Model ----------------

class CaiT_mod_ver1(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        cls_depth,
        heads,
        mlp_dim,
        dim_head = 64,
        dropout = 0.,
        emb_dropout = 0.,
        layer_dropout = 0.
    ):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2

        # patch embedding
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                      p1 = patch_size, p2 = patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        )

        # positional + class token
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.cls_token     = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout       = nn.Dropout(emb_dropout)

        # two-stage transformer
        self.patch_transformer = Transformer(dim, depth, heads, dim_head,
                                             mlp_dim, dropout, layer_dropout)
        self.cls_transformer   = Transformer(dim, cls_depth, heads, dim_head,
                                             mlp_dim, dropout, layer_dropout)

        # classification head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        # 1) embed patches
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        # 2) add positional embeddings & dropout
        x = x + self.pos_embedding[:, :n]
        x = self.dropout(x)

        # 3) patch transformer
        x = self.patch_transformer(x)

        # 4) prepend cls tokens for classification transformer
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = self.cls_transformer(cls_tokens, context = x)

        # 5) classification head
        return self.mlp_head(x[:, 0])

