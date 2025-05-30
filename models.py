"""Custom code implementing a transformer.
"""

import torch
import torch.nn as nn
from torch import nn as nn
from torch.nn import functional as F, TransformerEncoderLayer, TransformerEncoder
import math
from pos_encodings import RotaryPositionalEmbedding, apply_rotary_pos_emb


def get_sinusoidal_positional_embeddings_2(seq_len, d_model):
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)
    return pe


def get_sinusoidal_positional_embeddings(n_pos, dim, time=10000.0):
    # Create the denominator based on the dimension
    denom = torch.pow(time, 2 * (torch.arange(dim) // 2).float() / dim)

    # Create the positional encoding matrix
    position_enc = torch.arange(n_pos).unsqueeze(1).float() / denom.unsqueeze(0)

    # Apply the sine and cosine functions to the positional encoding matrix
    position_enc = torch.cat([torch.sin(position_enc[:, 0::2]), torch.cos(position_enc[:, 1::2])], dim=-1)

    return position_enc.unsqueeze(0)


class MaskedCausalAttention(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super().__init__()

        self.h_dim = h_dim

        self.n_heads = n_heads
        self.max_T = max_T

        self.q_net = nn.Linear(h_dim, h_dim)  # note, embedding dim = h_dim for now
        self.k_net = nn.Linear(h_dim, h_dim)
        self.v_net = nn.Linear(h_dim, h_dim)

        self.proj_net = nn.Linear(h_dim, h_dim)

        self.drop_p = drop_p
        self.att_drop = nn.Dropout(drop_p)
        self.proj_drop = nn.Dropout(drop_p)

        ones = torch.ones((max_T, max_T))
        mask = torch.tril(ones).view(1, 1, max_T, max_T)

        # register buffer makes sure mask does not get updated
        # during backpropagation
        self.register_buffer('mask', mask)

        # Custom initialization
        self.init_weights()
        self.training = True
        self.rotary = RotaryPositionalEmbedding(h_dim//n_heads)

    def init_weights(self):
        # Initialize weights according to Reddy's code (note: weight initialization doesn't ssem to matter much )
        scale = 1 / math.sqrt(self.q_net.in_features)
        for layer in [self.q_net, self.k_net, self.v_net, self.proj_net]:
            nn.init.normal_(layer.weight, mean=0.0, std=scale)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

    def forward(self, x, save_weights=False):
        B, T, C = x.shape  # batch size, seq length, h_dim * n_heads

        N, D = self.n_heads, C // self.n_heads  # N = num heads, D = attention dim

        # rearrange q, k, v as (B, N, T, D)
        q = self.q_net(x).view(B, T, N, D).transpose(1, 2)
        k = self.k_net(x).view(B, T, N, D).transpose(1, 2)
        v = self.v_net(x).view(B, T, N, D).transpose(1, 2)

        # calculate rotary position embeddings
        cos, sin = self.rotary(x, seq_dim=1)  # cos, sin shapes: (T, D/2)
        # Expand cos and sin across batch and heads to match q and k shapes
        cos = cos.transpose(0, 1).transpose(1, 2)  # shape now [batch, n_heads, seq_len, dim_head]
        sin = sin.transpose(0, 1).transpose(1, 2)  # shape now [batch, n_heads, seq_len, dim_head]

        # apply rotary position embeddings to queries and keys
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # weights (B, N, T, T)
        weights = q @ k.transpose(2, 3) / math.sqrt(D)
        # causal mask applied to weights
        weights = weights.masked_fill(self.mask[..., :T, :T] == 0, float('-inf'))
        # normalize weights, all -inf -> 0 after softmax
        normalized_weights = F.softmax(weights, dim=-1)

        # attention (B, N, T, D)
        attention = self.att_drop(normalized_weights @ v)  # multiply weights by values and apply dropout

        # gather heads and project (B, N, T, D) -> (B, T, N*D)
        attention = attention.transpose(1, 2).contiguous().view(B, T, N * D)

        out = self.proj_net(attention)
        if self.drop_p:
            out = self.proj_drop(out)
        if save_weights:
            return out, normalized_weights.detach()
        return out, None


class Block(nn.Module):
    """Transformer block. Consists of masked causal attention and a feedforward layer.

    Note: we use layernorms both after the attention and after the feedforward layer.
    """

    def __init__(self, h_dim, max_T, n_heads, drop_p, widening_factor=4, include_mlp=True, apply_ln=False,
                 activation='relu'):
        super().__init__()

        self.include_mlp = include_mlp
        self.apply_ln = apply_ln

        if activation == 'relu':
            self.activation = nn.ReLU
        elif activation == 'gelu':
            self.activation = nn.GELU
        else:
            raise ValueError("activation must be 'relu' or 'gelu'.")

        self.attention = MaskedCausalAttention(h_dim, max_T, n_heads, drop_p)

        self.mlp = nn.Sequential(
            nn.Linear(h_dim, widening_factor * h_dim),
            self.activation(),
            nn.Dropout(drop_p),
            nn.Linear(widening_factor * h_dim, h_dim),
        )

        self.ln1 = nn.LayerNorm(h_dim)
        self.ln2 = nn.LayerNorm(h_dim)

    def forward(self, x, index=None, save_weights=False):
        out_dict = {}
        # Attention -> LayerNorm -> MLP -> LayerNorm
        attention, weights = self.attention(x, save_weights=save_weights)
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
    def __init__(self, n_blocks=None, h_dim=None, max_T=None, n_heads=None, drop_p=None,
                 widening_factor=4, config=None, out_dim=None, include_mlp=True, apply_ln=False, input_embedder=None,
                 emb_dim=0, pos_embed_type='parity', pos_emb_loc='add'):
        super().__init__()

        if config:
            n_blocks = config.n_blocks
            h_dim = config.h_dim
            emb_dim = config.emb_dim
            pos_dim = config.pos_dim
            max_T = config.max_T
            n_heads = config.n_heads
            drop_p = config.drop_p
            widening_factor = config.widening_factor
            out_dim = config.out_dim
            include_mlp = config.include_mlp
            apply_ln = config.apply_ln
            pos_embed_type = config.pos_emb_type
            pos_emb_loc = config.pos_emb_loc
        elif None in [n_blocks, h_dim, max_T, n_heads, drop_p]:
            raise ValueError("Either provide a complete config or all hyperparameters individually.")

        self.input_embedder = input_embedder

        if pos_emb_loc == 'append' or pos_emb_loc == 'none':
            h_dim = emb_dim + pos_dim
        if pos_emb_loc == 'add' and emb_dim != h_dim:
            raise ValueError("If pos_emb_loc is 'add', emb_dim must be equal to h_dim.")

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
                                           include_mlp=include_mlp[b], apply_ln=apply_ln) for b in range(n_blocks)])

        # projection head
        self.ln = nn.LayerNorm(h_dim)
        self.proj_head = nn.Linear(h_dim, out_dim)
        # position embedding
        self.positional_embedding = get_sinusoidal_positional_embeddings_2(max_T, emb_dim)
        self.pos_emb_loc = pos_emb_loc  # add or append to token embeddings

    def forward(self, x, save_weights=False, save_hidden_activations=False, apply_embedder=True):
        out_dict = {}
        hidden_activations = []
        # embed inputs, if required
        if self.input_embedder is None or not apply_embedder:
            h = x
        else:
            h = self.input_embedder(x)
        # apply positional encoding
        if self.pos_emb_loc == 'add':
            h = h + self.positional_embedding[:, :h.shape[1], :]
        elif self.pos_emb_loc == 'append':
            batch_size, seq_len, _ = h.shape
            # Expand the positional embeddings to match the batch size
            positional_embeddings = self.positional_embedding[:, :seq_len, :].expand(batch_size, -1, -1)
            # Concatenate the embeddings
            h = torch.cat([h, positional_embeddings], dim=-1)
        elif self.pos_emb_loc == 'none':
            pass
        else:
            raise ValueError("pos_emb_loc must be 'add' or 'append'.")
        # pass through the transformer layers
        for index, block in enumerate(self.blocks):
            h, out = block(h, index=index, save_weights=save_weights)
            if save_weights:
                out_dict[f'block_{index}'] = out
            if save_hidden_activations:
                hidden_activations.append(h.clone())  # Save a copy of the hidden activation

        if save_hidden_activations:
            out_dict['hidden_activations'] = hidden_activations

        # finally, predict the logits
        pred = self.proj_head(h)

        return pred[:, -1], out_dict


if __name__=='__main__':
    mod = MaskedCausalAttention(128, 32, 1, .1)


class MyTransformer(nn.Module):
    """Standard transformer model (encoder only).
    """
    def __init__(self, config, device):
        super(MyTransformer, self).__init__()
        self.encoder_layers = TransformerEncoderLayer(d_model=config.model.emb_dim * 2, nhead=config.model.n_heads)
        self.transformer_encoder = TransformerEncoder(encoder_layer=self.encoder_layers,
                                                      num_layers=config.model.n_blocks).to(device)
        self.linear = nn.Linear(config.model.emb_dim * 2, 1).to(device)

    def forward(self, src):
        output = self.transformer_encoder(src)
        output = self.linear(output)
        return output
