"""Real-valued feature-then-time InterFormer, derived from InterCVT."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class InterFormerNorm(nn.Module):
    """Preserve the original L2 normalization and dim**-0.5 scale."""
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** -0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.gamma * self.scale


class ReLUWithBias(nn.Module):
    """Real activation; replaces the original nonstandard complex ModReLU."""
    def __init__(self, squared=True):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(()))
        self.squared = squared

    def forward(self, x):
        x = F.relu(x + self.bias)
        return x.square() if self.squared else x


def attention_weights(q, k, dis=False, feature_axis=False):
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    # Match the distinct formulas in intercross_attend.py and attend.py.
    if feature_axis and dis:
        scores = -scores
    weights = scores.softmax(dim=-1)
    if dis and not feature_axis:
        weights = 1 - weights
    return weights


class FeatureAttention(nn.Module):
    def __init__(self, dim, time_depth, heads, dim_head, dis=False):
        super().__init__()
        self.heads, self.dim_head, self.dim = heads, dim_head, dim
        self.dis = dis
        self.to_q = nn.Linear(time_depth, heads * dim_head, bias=False)
        self.to_k = nn.Linear(time_depth, heads * dim_head, bias=False)
        self.to_v = nn.Linear(dim, heads * dim, bias=False)
        self.to_out = nn.Linear(heads * dim, dim, bias=False)

    def forward(self, x):
        b, t, d = x.shape
        features = x.transpose(1, 2)
        q = self.to_q(features).reshape(b, d, self.heads, self.dim_head).transpose(1, 2)
        k = self.to_k(features).reshape(b, d, self.heads, self.dim_head).transpose(1, 2)
        v = self.to_v(x).reshape(b, t, self.heads, d).transpose(1, 2)
        weights = attention_weights(q, k, self.dis, feature_axis=True)
        out = (v @ weights).transpose(1, 2).reshape(b, t, self.heads * d)
        return self.to_out(out)


class TimeAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, dis=True):
        super().__init__()
        self.heads, self.dim_head, self.dis = heads, dim_head, dis
        inner = heads * dim_head
        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_kv = nn.Linear(dim, inner * 2, bias=False)
        self.to_out = nn.Linear(inner, dim, bias=False)

    def forward(self, x):
        b, t, _ = x.shape
        q, k, v = self.to_q(x), *self.to_kv(x).chunk(2, dim=-1)
        q, k, v = [z.reshape(b, t, self.heads, self.dim_head).transpose(1, 2)
                   for z in (q, k, v)]
        out = attention_weights(q, k, self.dis) @ v
        return self.to_out(out.transpose(1, 2).reshape(b, t, -1))


class AttentionBlock(nn.Module):
    def __init__(self, dim, attention, ff_mult, squared, dropout):
        super().__init__()
        self.attn_norm = InterFormerNorm(dim)
        self.attn = attention
        self.ff_norm = InterFormerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * ff_mult), ReLUWithBias(squared),
                                nn.Linear(dim * ff_mult, dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        return self.dropout(x + self.ff(self.ff_norm(x)))


class InterFormer(nn.Module):
    """Input [batch, time_depth, dim] of real GRD+GLCM; output raw [batch, 2] logits.

    Only the user's part2=True architecture is retained. No LSTM, complex dtype,
    inverse architecture, unused embedding or inactive classification heads.
    """
    def __init__(self, dim, *, time_depth=11, num_tokens=32, dim_head=64, heads=8,
                 dattn_depth=1, tattn_depth=1, part2=True, dis=False, t_dis=True,
                 ff_mult=4, relu_squared=True, dropout=0.2):
        super().__init__()
        if part2 is not True:
            raise ValueError('InterFormer retains only the part2=True architecture.')
        if min(dim, time_depth, num_tokens, dim_head, heads, dattn_depth, tattn_depth, ff_mult) < 1:
            raise ValueError('Dimensions and depths must be positive.')
        self.config = dict(dim=dim, time_depth=time_depth, num_tokens=num_tokens,
                           dim_head=dim_head, heads=heads, dattn_depth=dattn_depth,
                           tattn_depth=tattn_depth, part2=True, dis=dis, t_dis=t_dis,
                           ff_mult=ff_mult, relu_squared=relu_squared, dropout=dropout)
        self.dim, self.time_depth = dim, time_depth
        # Same frequencies as original for even dimensions; also support odd dim.
        half = max(dim // 2, 1)
        frequencies = 10000 ** (torch.arange((dim + 1) // 2).float() / half)
        angles = torch.arange(time_depth).float()[:, None] / frequencies
        pos = torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(1)[:, :dim]
        self.register_buffer('position', pos.unsqueeze(0))
        self.dim_layers = nn.ModuleList([
            AttentionBlock(dim, FeatureAttention(dim, time_depth, heads, dim_head, dis),
                           ff_mult, relu_squared, dropout) for _ in range(dattn_depth)])
        self.time_layers = nn.ModuleList([
            AttentionBlock(dim, TimeAttention(dim, heads, dim_head, t_dis),
                           ff_mult, relu_squared, dropout) for _ in range(tattn_depth)])
        self.norm = InterFormerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Flatten(1), nn.Linear(dim * time_depth, num_tokens * 2), nn.ReLU(),
            nn.Linear(num_tokens * 2, num_tokens * 4), nn.ReLU(),
            nn.Linear(num_tokens * 4, 2))

    def forward(self, x):
        if x.is_complex() or not x.is_floating_point():
            raise TypeError('Expected real floating-point GRD+GLCM input.')
        if x.ndim != 3 or tuple(x.shape[1:]) != (self.time_depth, self.dim):
            raise ValueError(f'Expected [batch, {self.time_depth}, {self.dim}], got {tuple(x.shape)}')
        x = x + self.position
        for block in self.dim_layers:
            x = block(x)
        for block in self.time_layers:
            x = block(x)
        # Raw signed logits: CrossEntropyLoss handles the softmax itself.
        return self.classifier(self.dropout(self.norm(x)))
