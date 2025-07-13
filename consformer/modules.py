import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
import math


class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, expand_size: int, act: nn.Module = nn.GELU,
                 drop: float = 0.1, bias: bool = True):
        super().__init__()

        self.fc1 = nn.Linear(hidden_size, expand_size, bias=bias)
        self.act = act()
        self.fc2 = nn.Linear(expand_size, hidden_size, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_size, hidden_size, head_count=8, dropout=0, normalize_energy=False, rpe="no"):
        super().__init__()
        self.tf_hidden_size = hidden_size  # transformer hidden state size
        self.head_count = head_count  # number of attention heads

        self.Q = nn.Linear(embedding_size, hidden_size * head_count)
        self.K = nn.Linear(embedding_size, hidden_size * head_count)
        self.V = nn.Linear(embedding_size, hidden_size * head_count)

        self.drop_out = nn.Dropout(dropout)
        self.projection = nn.Linear(hidden_size * head_count, hidden_size)

        self.normalize_energy = normalize_energy
        self.rpe = rpe
        if rpe == "mask":
            self.rpe_c = float('-inf')
        elif rpe == "learned":
            self.rpe_c = nn.Parameter(torch.ones(1))
        else:
            self.rpe_c = 0

    def forward(self, x, rpe=None):
        # input x has the shape: (batch_size, sequence_len, token_embedding_size)

        batch_size, seq_len, _ = x.size()

        # Q(x), K(x), V(x) has shape (batch, sequence_len,  num_heads * transformer_hidden_size)
        # after reshape, qkv all have size (batch x head_count x seq_length x hidden_size)
        queries = self.reshape_qkv(self.Q(x), batch_size, seq_len)
        keys = self.reshape_qkv(self.K(x), batch_size, seq_len)
        values = self.reshape_qkv(self.V(x), batch_size, seq_len)

        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)

        # scaled dot product attention
        energy = energy / math.sqrt(self.tf_hidden_size)

        # Masking parts of the sequence
        if rpe is not None:
            if self.rpe == "mask":
                if rpe.dim() == 2:
                    energy = energy.masked_fill(rpe.view(batch_size, 1, 1, seq_len), float('-inf'))
                elif rpe.dim() == 3:
                    energy = energy.masked_fill(rpe.view(batch_size, 1, seq_len, seq_len), float('-inf'))
            else:
                if rpe.dim() == 2:
                    energy = energy + rpe.view(batch_size, 1, 1, seq_len) * self.rpe_c
                elif rpe.dim() == 3:
                    energy = energy + rpe.view(batch_size, 1, seq_len, seq_len) * self.rpe_c

        attention = F.softmax(energy, dim=-1)  # softmax over k for each q
        attention = self.drop_out(attention)

        out = torch.einsum('bhal, bhld -> bhad', attention, values)
        out = out.permute(0, 2, 1, 3)  # [Batch, Seq_len, Head, Hidden_dims]
        out = out.reshape(batch_size, seq_len, self.tf_hidden_size * self.head_count)
        out = self.projection(out)  # [batch_size, seq_len]

        return out

    def reshape_qkv(self, x, batch_size, seq_length):
        # first reshape (batch, sequence_len,  num_heads * transformer_hidden_size) to (batch, sequence_len,  num_heads, transformer_hidden_size)
        x = x.reshape(batch_size, seq_length, self.head_count, self.tf_hidden_size)
        # then permute to get size (batch, num_heads,  sequence_len, transformer_hidden_size)
        x = x.permute(0, 2, 1, 3)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embedding_size: int, hidden_size: int, num_heads: int, expand_size: int,
                 attention: nn.Module = MultiHeadAttention, act: nn.Module = nn.GELU,
                 attn_drop: float = 0.1, ffn_drop: float = 0.1,
                 bias: bool = True, rpe="no"):
        super().__init__()
        # first pre-norm layer
        self.norm1 = nn.LayerNorm(hidden_size)
        # initialize the attention layer
        self.attn = attention(
            embedding_size=embedding_size, hidden_size=hidden_size, head_count=num_heads, dropout=attn_drop, rpe=rpe
        )

        # second pre-norm layer
        self.norm2 = nn.LayerNorm(hidden_size)
        # initialize the feed forward network (MLP)
        self.ffn = FeedForward(
            hidden_size=hidden_size, expand_size=expand_size, act=act,
            drop=ffn_drop, bias=bias,
        )

    def forward(self, x: Tensor, rpe=None):
        # normalize input then add residual to attention output
        x = x + self.attn(self.norm1(x), rpe)
        # normalize input then add residual to feedforward output
        return x + self.ffn(self.norm2(x))
