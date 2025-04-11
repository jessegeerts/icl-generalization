"""Custom code implementing a transformer.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import math


class MaskedCausalAttention(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p, softmax_attn=True, w_init_scale=None):
        super().__init__()

        self.h_dim = h_dim

        self.n_heads = n_heads
        self.max_T = max_T

        self.q_net = nn.Linear(h_dim, h_dim)
        self.k_net = nn.Linear(h_dim, h_dim)
        self.v_net = nn.Linear(h_dim, h_dim)

        if w_init_scale is None:
            self.w_init_scale = 1 / math.sqrt(self.q_net.in_features)
        else:
            self.w_init_scale = w_init_scale

        self.proj_net = nn.Linear(h_dim, h_dim)

        self.drop_p = drop_p
        self.att_drop = nn.Dropout(drop_p)
        self.proj_drop = nn.Dropout(drop_p)
        self.softmax_attn = softmax_attn

        ones = torch.ones((max_T, max_T))
        mask = torch.tril(ones).view(1, 1, max_T, max_T)

        # register buffer makes sure mask does not get updated
        # during backpropagation
        self.register_buffer('mask', mask)

        # Custom initialization
        self.init_weights()
        self.training = True

    def init_weights(self):
        # Initialize weights according to Reddy's code (note: weight initialization doesn't ssem to matter much )
        scale = self.w_init_scale
        for layer in [self.q_net, self.k_net, self.v_net, self.proj_net]:
            nn.init.normal_(layer.weight, mean=0.0, std=scale)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

    def forward(self, x, save_weights=False, head_mask=None, include_query=True):

        B, T, C = x.shape  # batch size, seq length, h_dim * n_heads

        if include_query:
            key_dim = T
        else:
            key_dim = T - 1

        N, D = self.n_heads, C // self.n_heads  # N = num heads, D = attention dim

        # rearrange q, k, v as (B, N, T, D)
        q = self.q_net(x).view(B, T, N, D).transpose(1, 2)
        # For keys and values, only use the first key_dim tokens if not including query
        k = self.k_net(x[:, :key_dim]).view(B, key_dim, N, D).transpose(1, 2)
        v = self.v_net(x[:, :key_dim]).view(B, key_dim, N, D).transpose(1, 2)

        # weights (B, N, T, key_dim)
        weights = q @ k.transpose(2, 3) / math.sqrt(D)

        if self.softmax_attn:
            # Original approach with softmax
            weights = weights.masked_fill(self.mask[..., :T, :key_dim] == 0, float('-inf'))
            normalized_weights = F.softmax(weights, dim=-1)
        else:
            # Linear attention approach - use 0 for masked positions
            normalized_weights = weights * self.mask[..., :T, :key_dim]

        # attention (B, N, T, D)
        attention = self.att_drop(normalized_weights @ v)  # multiply weights by values and apply dropout
        attention = torch.clamp(attention, min=-10.0, max=10.0)

        if head_mask is not None:
            # Convert head_mask to proper tensor shape for broadcasting
            mask_tensor = torch.tensor(head_mask, device=attention.device)
            # Reshape to [1, N, 1, 1] for broadcasting across batch and sequence dimensions
            mask_tensor = mask_tensor.view(1, N, 1, 1)
            # Apply mask (multiply by 0 or 1)
            attention = attention * mask_tensor

        # gather heads and project (B, N, T, D) -> (B, T, N*D)
        attention = attention.transpose(1, 2).contiguous().view(B, T, N * D)

        out = self.proj_net(attention)
        if self.drop_p:
            out = self.proj_drop(out)
        out = torch.clamp(out, min=-10.0, max=10.0)
        if save_weights:
            return out, normalized_weights.detach()
        return out, None


class Block(nn.Module):
    """Transformer block. Consists of masked causal attention and a feedforward layer.

    Note: we use layernorms both after the attention and after the feedforward layer.
    """

    def __init__(self, h_dim, max_T, n_heads, drop_p, widening_factor=4, include_mlp=True, apply_ln=False,
                 activation='relu', softmax_attn=True, w_init_scale=None):
        super().__init__()

        self.include_mlp = include_mlp
        self.apply_ln = apply_ln

        if activation == 'relu':
            self.activation = nn.ReLU
        elif activation == 'gelu':
            self.activation = nn.GELU
        else:
            raise ValueError("activation must be 'relu' or 'gelu'.")

        self.attention = MaskedCausalAttention(h_dim, max_T, n_heads, drop_p, softmax_attn=softmax_attn, w_init_scale=w_init_scale)

        self.mlp = nn.Sequential(
            nn.Linear(h_dim, widening_factor * h_dim),
            self.activation(),
            nn.Dropout(drop_p),
            nn.Linear(widening_factor * h_dim, h_dim),
        )

        self.ln1 = nn.LayerNorm(h_dim)
        self.ln2 = nn.LayerNorm(h_dim)

    def forward(self, x, index=None, save_weights=False, head_mask=None, include_query=True):
        out_dict = {}
        # Attention -> LayerNorm -> MLP -> LayerNorm
        attention, weights = self.attention(x, save_weights=save_weights, head_mask=head_mask, include_query=include_query)
        if save_weights:
            out_dict['weights'] = weights
        x = x + attention  # residual
        if self.apply_ln:
            x = self.ln1(x)
        if self.include_mlp:
            x = x + self.mlp(x)  # residual
        if self.apply_ln:
            x = self.ln2(x)
        return x, out_dict


class Transformer(nn.Module):
    def __init__(self, n_blocks=None, h_dim=None, max_T=None, n_heads=None, drop_p=None, w_init_scale=None,
                 widening_factor=4, config=None, out_dim=None, include_mlp=True, apply_ln=False, softmax_attn=True):
        super().__init__()

        if config:
            n_blocks = config.n_blocks
            h_dim = config.h_dim
            max_T = config.max_T
            n_heads = config.n_heads
            drop_p = config.drop_p
            widening_factor = config.widening_factor
            out_dim = config.out_dim
            include_mlp = config.include_mlp
            apply_ln = config.apply_ln
            softmax_attn = config.softmax_attn
            w_init_scale = config.w_init_scale
        elif None in [n_blocks, h_dim, max_T, n_heads, drop_p]:
            raise ValueError("Either provide a complete config or all hyperparameters individually.")

        self.input_embedder = None  # for potential future use.

        if isinstance(softmax_attn, bool):
            softmax_attn = [softmax_attn] * n_blocks
        else:
            assert len(softmax_attn) == n_blocks, "softmax_attn must be a list of size n_blocks."

        # determine which blocks include an MLP classifier
        if include_mlp is True:
            include_mlp = [True] * n_blocks
        elif include_mlp is False:
            include_mlp = [False] * n_blocks
        else:
            assert len(include_mlp) == n_blocks, "include_mlp must be a list of size n_blocks."

        # transformer blocks
        self.blocks = nn.ModuleList([Block(h_dim, max_T, n_heads, drop_p,
                                           widening_factor=widening_factor,
                                           include_mlp=include_mlp[b], apply_ln=apply_ln,
                                           softmax_attn=softmax_attn[b],
                                           w_init_scale=w_init_scale) for b in range(n_blocks)])

        # projection head
        self.ln = nn.LayerNorm(h_dim)
        self.proj_head = nn.Linear(h_dim, out_dim)

    def forward(self, x, save_weights=False, head_mask=None, include_query=True):
        out_dict = {}
        # embed inputs, if required
        if self.input_embedder is None:
            h = x
        else:
            h = self.input_embedder(x)
        # pass through the transformer layers
        for index, block in enumerate(self.blocks):
            h, out = block(h, index=index, save_weights=save_weights, head_mask=head_mask, include_query=include_query)
            if save_weights:
                out_dict[f'block_{index}'] = out
        # finally, predict the logits
        pred = self.proj_head(h)

        return pred[:, -1], out_dict


if __name__=='__main__':
    mod = MaskedCausalAttention(128, 32, 1, .1)
