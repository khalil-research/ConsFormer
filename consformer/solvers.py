from .modules import *
from .embeddings import *
from .destroy import RandomDestroyStrategy


class ConsFormer(nn.Module):

    def __init__(self,
                 input_size: int,
                 embedding_size: int,
                 hidden_size: int,
                 num_heads: int,
                 expand_size: int,
                 output_size: int = 1,
                 attention: nn.Module = MultiHeadAttention,
                 act: nn.Module = nn.GELU,
                 drop_out: float = 0,
                 bias: bool = True,
                 num_layers=1,
                 vocab_size=None,
                 subset_threshold=0,
                 ape_dim=1,
                 mixing_strategy="add",
                 tau=0.1,
                 rpe="mask",
                 greedy_decode=False,
                 task="sudoku",
                 destroy_strategy=None,
                 ):

        super().__init__()

        self.hidden_size = hidden_size

        self.mask_embedding = nn.Parameter(torch.empty(embedding_size).uniform_(-1, 1))
        self.embedding_layer = EmbeddingLayer(embedding_size, vocab_size=vocab_size)

        if ape_dim > 0:
            self.positional_encoding = FixedAbsolutePositionalEmbedding(embedding_size // ape_dim)
        elif ape_dim == 0:
            self.positional_encoding = None

        self.embedding_mixer = EmbeddingMixer(embedding_size, mixing_strategy)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(embedding_size=embedding_size,
                                                                  hidden_size=hidden_size, num_heads=num_heads,
                                                                  expand_size=expand_size,
                                                                  attention=attention, act=act, bias=bias,
                                                                  attn_drop=drop_out, ffn_drop=drop_out, rpe=rpe) for _
                                                 in range(num_layers)])
        self.classifier = nn.Linear(hidden_size, 1)
        self.head = nn.Linear(hidden_size, output_size)

        self.subset_improvement = True if subset_threshold > 0 else False
        self.subset_threshold = subset_threshold
        self.ape_dim = ape_dim
        self.tau = tau
        self.greedy_decode = greedy_decode

        self.task = task
        self.destroy_strategy = destroy_strategy if destroy_strategy is not None else RandomDestroyStrategy()


    def _get_mask_indices(self, x: Tensor):
        """
        x has size (batch_size, seq_len, embedding_size)
        we want a mask of size (batch_size, seq_len) representing the items need to be masked (ie the variables not yet assigned)
        in the tensor, they are represented by a embedding consisting only of zeros
        """
        mask = torch.sum(x, dim=-1) == 0  # check if none of the elements is 1 (i.e., sum is 0)
        return mask

    def _sample_subset(self, inputs):
        """
        Delegate subset sampling to the configured destroy strategy.
        Returns a boolean mask of editable variables to update.
        """

        updated_var_inds = self.destroy_strategy.make_selection(self.subset_threshold, inputs)

        return updated_var_inds

    def _decode(self, logits):

        if self.greedy_decode:
            scaled_logits = logits / self.tau
            predictions = F.softmax(scaled_logits, dim=-1)
        else:
            predictions = F.gumbel_softmax(logits, tau=self.tau, hard=False, dim=-1)

        return predictions

    def _get_pos_embedding(self, x):
        """
        get the absolute positional embedding. Currently hard-coded for the implemented tasks
        (dim 1 for graph coloring and max cut, dim 2 for sudoku, dim 3 for nurse rostering).
        """

        batch_size = x.size(0)

        if self.ape_dim == 1:
            seq_len = x.size(1)
            position_ids = torch.arange(seq_len, device=x.device)
            position_ids = position_ids.repeat(batch_size, 1)
            pos = self.positional_encoding(position_ids)

        elif self.ape_dim == 2:
            # currently hard coded for sudoku
            position_ids_x = torch.arange(9, device=x.device).repeat(9)
            position_ids_y = torch.arange(9, device=x.device).repeat_interleave(9)

            position_ids_x = position_ids_x.repeat(batch_size, 1)
            position_ids_y = position_ids_y.repeat(batch_size, 1)

            position_embeds_x = self.positional_encoding(position_ids_x)
            position_embeds_y = self.positional_encoding(position_ids_y)

            pos = torch.cat((position_embeds_x, position_embeds_y), dim=-1)

        elif self.ape_dim == 3:
            # currently hard coded for nurse rostering
            batch_size, var_count, num_nurses = x.size()
            shifts_per_day, nurses_per_shift = 3, 3
            num_days = var_count // (shifts_per_day * nurses_per_shift)

            position_ids_x = torch.arange(num_days, device=x.device).repeat(shifts_per_day * nurses_per_shift)
            position_ids_y = torch.arange(shifts_per_day, device=x.device).repeat_interleave(num_days).repeat(
                nurses_per_shift)
            position_ids_z = torch.arange(nurses_per_shift, device=x.device).repeat_interleave(
                num_days * shifts_per_day)

            position_ids_x = position_ids_x.repeat(batch_size, 1)
            position_ids_y = position_ids_y.repeat(batch_size, 1)
            position_ids_z = position_ids_z.repeat(batch_size, 1)

            position_embeds_x = self.positional_encoding(position_ids_x)
            position_embeds_y = self.positional_encoding(position_ids_y)
            position_embeds_z = self.positional_encoding(position_ids_z)

            pos = torch.cat((position_embeds_x, position_embeds_y, position_embeds_z), dim=-1)

        elif self.ape_dim == 0:
            pos = None

        return pos


    def forward(self, inputs):
        """
        inputs:
          - soft_onehot: (B, S, V)
          - logits: (B, S, V)
          - binary_constraint_graph: (B, S, S)
          - variable_ind: (B, S) bool
        """
        x = inputs["soft_onehot"]
        logits = inputs["logits"]
        binary_constraint_graph = inputs["binary_constraint_graph"]
        variable_ind = inputs["variable_ind"]
        batch_size, seq_len, embedding_size = x.size()
        input_raw = x.clone()

        x = self.embedding_layer(x)
        pos = self._get_pos_embedding(x)
        if self.subset_improvement:
            variable_ind = self._sample_subset(inputs)
        x = self.embedding_mixer(x, self.mask_embedding, pos, variable_ind)

        rpe = ~binary_constraint_graph.bool()
        for block in self.transformer_blocks:
            x = block(x, rpe)

        out_logits = self.head(x)

        predictions = self._decode(out_logits)

        variable_ind_expanded = variable_ind.unsqueeze(-1).expand_as(input_raw)
        final_pred_onehot = torch.where(variable_ind_expanded == 1, predictions, input_raw)

        return {"preds_onehot":final_pred_onehot, "out_logits":out_logits}
