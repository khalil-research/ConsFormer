import torch
import torch.nn as nn


class EmbeddingLayer(nn.Module):
    def __init__(self, embedding_size, vocab_size=None):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embedding_size)

    def forward(self, x):
        batch_size, seq_len, input_size = x.size()
        if input_size != 1:  # if input is represented as one-hot, retrieve the index
            x_index = torch.argmax(x, dim=-1)
        else:
            x_index = x.squeeze(-1)  # change (batch_size, seq_len, 1) to just (batch_size, seq_len)
        x = self.embedding_layer(x_index)  # outputs (batch_size, seq_len, input_size
        return x


class FixedAbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(16384).type_as(inv_freq)
        sinusoid_inp = torch.einsum("i , j -> i j", t, inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        self.embed = nn.Embedding.from_pretrained(emb, freeze=True)

    def forward(self, position_ids: torch.Tensor):
        return self.embed(position_ids.long())


class EmbeddingMixer(nn.Module):
    def __init__(self, embed_dim=None, mix_strategy=None):
        super(EmbeddingMixer, self).__init__()
        self.embed_dim = embed_dim
        self.mix_strategy = mix_strategy

        self.token_weight = nn.Parameter(torch.ones(1))
        self.mask_weight = nn.Parameter(torch.ones(1))
        self.position_weight = nn.Parameter(torch.ones(1))

        if mix_strategy == "concat":
            self.token_project = nn.Linear(embed_dim, embed_dim // 2)
            self.pe_project = nn.Linear(embed_dim, embed_dim // 2)

    def forward(self, token_embeds, mask_embeds, position_embeds, mask_inds=None):
        """
        token_embeds: (batch_size, seq_len, embed_dim)
        mask_embeds: (embed_dim)
        position_embeds: (batch_size, seq_len, embed_dim)
        """

        if mask_inds is None:
            mask_inds = torch.ones(token_embeds.shape[0], token_embeds.shape[1], dtype=torch.bool)

        if position_embeds is not None:
            if self.mix_strategy == "no":
                output_embeds = token_embeds + position_embeds
                output_embeds[mask_inds] = output_embeds[mask_inds] + mask_embeds
            else:
                output_embeds = (self.token_weight * token_embeds) + (self.position_weight * position_embeds)
                output_embeds[mask_inds] = output_embeds[mask_inds] + (self.mask_weight * mask_embeds)
        else:
            if self.mix_strategy == "no":
                output_embeds = token_embeds
                output_embeds[mask_inds] = token_embeds[mask_inds] + mask_embeds
            else:
                output_embeds = (self.token_weight * token_embeds)
                output_embeds[mask_inds] = output_embeds[mask_inds] + (self.mask_weight * mask_embeds)
        return output_embeds
