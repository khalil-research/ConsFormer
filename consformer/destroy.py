import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod

class DestroyStrategy(ABC):
    """Base interface for destroy operators."""

    def __init__(self, name, requires_grad=False):
        self.name = name
        self.requires_grad = requires_grad

    @abstractmethod
    def make_selection(self, **kwargs):
        """Return a boolean mask of selected variables."""
        raise NotImplementedError

    def get_ks(self, top_prob, variable_ind):
        # Determine how many variables to keep per instance based on the ratio.
        editable = variable_ind.sum(dim=-1)  # [B]
        k_per = (top_prob * editable).floor().to(torch.long)  # [B]
        k_max = int(k_per.max().item())

        return k_per, k_max

    def get_selected(self, variable_ind, scores, k_per, k_max):
        _, idx = torch.topk(scores, k=k_max, dim=1, largest=True, sorted=True)
        B, N = variable_ind.shape
        keep = torch.arange(k_max, device=variable_ind.device).expand(B, k_max) < k_per.unsqueeze(1)  # [B, k_max]
        selected = torch.zeros(B, N, dtype=torch.bool, device=variable_ind.device)  # [B, N]
        selected.scatter_(1, idx, keep)  # mark kept indices
        return selected

class RandomDestroyStrategy(DestroyStrategy):

    def __init__(self):
        super(RandomDestroyStrategy, self).__init__(name='random')

    def make_selection(self, threshold, inputs):
        """
        Uniform random removal.
        """
        variable_ind = inputs["variable_ind"]

        # Generate Uniform noise with the same shape as var_inds
        noise = torch.rand(variable_ind.shape, device=variable_ind.device)
        mask = (noise > threshold)  # Noise greater than threshold will remain "True" in mask

        # Update the var_inds by applying the mask
        return variable_ind & mask


class GreedyGradientDestroyStrategy(DestroyStrategy):

    def __init__(self, loss_function, use_constraint_graph=False):
        super(GreedyGradientDestroyStrategy, self).__init__(name='greedygradient')
        self.loss_function = loss_function
        self.use_constraint_graph = use_constraint_graph

    def make_selection(self, threshold, inputs):
        """
        Greedy gradient-guided removal from logits.
        """
        variable_ind = inputs["variable_ind"]
        logits = inputs["logits"]
        if self.use_constraint_graph:
            binary_graph = inputs["binary_constraint_graph"]

        top_prob = 1 - threshold

        # Determine how many variables to keep per instance based on the ratio.
        editable = variable_ind.sum(dim=-1)  # [B]
        k_per = (top_prob * editable).floor().to(torch.long)  # [B]
        k_max = int(k_per.max().item())
        if k_max == 0:
            return variable_ind

        # Treat mu as categorical probabilities before evaluating the loss.
        with torch.enable_grad():
            logits_clone = logits.clone().detach().requires_grad_(True)
            predictions = F.softmax(logits_clone, dim=-1)
            if self.use_constraint_graph:
                loss = self.loss_function(predictions, binary_graph)
            else:
                loss = self.loss_function(predictions)
            loss = loss.sum()
            gradients = torch.autograd.grad(loss, logits_clone, retain_graph=False, create_graph=False, allow_unused=False)[0]
            grad_scores = gradients.detach().norm(p=1, dim=-1)

        # don't care about the gradients for the unchangeable variables
        grad_scores = grad_scores.masked_fill(~variable_ind, float('-inf'))

        _, idx = torch.topk(grad_scores, k=k_max, dim=1, largest=True, sorted=True)
        # build row-wise keep mask: for row b, keep first k_per[b] indices
        B, N = variable_ind.shape
        device = variable_ind.device
        keep = torch.arange(k_max, device=device).expand(B, k_max) < k_per.unsqueeze(1)  # [B, k_max]

        selected = torch.zeros(B, N, dtype=torch.bool, device=device)  # [B, N]
        selected.scatter_(1, idx, keep)  # mark kept indices

        return variable_ind & selected


class StochasticGradientDestroyStrategy(DestroyStrategy):

    def __init__(self, loss_function, use_constraint_graph=False):
        super(StochasticGradientDestroyStrategy, self).__init__(name='stochasticgradient')
        self.loss_function = loss_function
        self.use_constraint_graph = use_constraint_graph

    def make_selection(self, threshold, inputs):
        """
        Stochastic gradient-guided removal from logits.
        """
        variable_ind = inputs["variable_ind"]
        logits = inputs["logits"]
        if self.use_constraint_graph:
            binary_graph = inputs["binary_constraint_graph"]

        top_prob = 1 - threshold
        alpha = 1
        eps = 1e-8

        # Treat mu as categorical probabilities before evaluating the loss.
        with torch.enable_grad():
            logits_clone = logits.clone().detach().requires_grad_(True)
            predictions = F.softmax(logits_clone, dim=-1)
            if self.use_constraint_graph:
                loss = self.loss_function(predictions, binary_graph)
            else:
                loss = self.loss_function(predictions)
            loss = loss.sum()
            gradients = torch.autograd.grad(loss, logits_clone, retain_graph=False, create_graph=False, allow_unused=False)[0]
            scores = gradients.detach().norm(p=1, dim=-1)

        mean = scores.mean(dim=-1, keepdim=True)
        std = scores.std(dim=-1, keepdim=True)
        scores_norm = (scores - mean) / (std + eps)

        probs = torch.sigmoid(alpha * scores_norm)  # bias by score
        avg_probs = probs.mean(dim=-1, keepdim=True)  # average over variables
        scale = top_prob / (avg_probs + eps)
        probs_scaled = (probs * scale).clamp(max=1.0)
        mask = torch.rand_like(probs_scaled) < probs_scaled

        return variable_ind & mask


class GreedyWorstDestroyStrategy(DestroyStrategy):

    def __init__(self, loss_function, use_constraint_graph=False):
        super(GreedyWorstDestroyStrategy, self).__init__(name='greedyworst')
        self.loss_function = loss_function
        self.use_constraint_graph = use_constraint_graph

    def make_selection(self, threshold, inputs):
        """
        Greedy worst removal using decoded assignments.
        """
        variable_ind = inputs["variable_ind"]
        x = inputs["soft_onehot"]
        if self.use_constraint_graph:
            binary_graph = inputs["binary_constraint_graph"]

        top_prob = 1 - threshold

        # Determine how many variables to keep per instance based on the ratio.
        k_per, k_max = self.get_ks(top_prob, variable_ind)
        if k_max <= 0:
            return variable_ind

        with torch.enable_grad():
            x_test = x.clone().detach()
            max_indices = torch.argmax(x_test, dim=-1, keepdim=True)
            predictions_one_hot = torch.zeros_like(x_test)
            predictions_one_hot = predictions_one_hot.scatter_(-1, max_indices, 1).requires_grad_(True)
            if self.use_constraint_graph:
                loss = self.loss_function(predictions_one_hot, binary_graph)
            else:
                loss = self.loss_function(predictions_one_hot)
            loss = loss.sum()
            gradients = torch.autograd.grad(loss, predictions_one_hot, retain_graph=False, create_graph=False,
                                            allow_unused=False)[0]
            scores = gradients.detach().norm(p=1, dim=-1)

        scores = scores.masked_fill(~variable_ind, float('-inf'))
        selected = self.get_selected(variable_ind, scores, k_per, k_max)

        return variable_ind & selected


class StochasticWorstDestroyStrategy(DestroyStrategy):

    def __init__(self, loss_function, use_constraint_graph=False):
        super(StochasticWorstDestroyStrategy, self).__init__(name='stochasticworst')
        self.loss_function = loss_function
        self.use_constraint_graph = use_constraint_graph

    def make_selection(self, threshold, inputs):
        """
        Stochastic worst removal using decoded assignments.
        """
        variable_ind = inputs["variable_ind"]
        x = inputs["soft_onehot"]
        if self.use_constraint_graph:
            binary_graph = inputs["binary_constraint_graph"]

        top_prob = 1 - threshold
        alpha = 1
        eps = 1e-8

        with torch.enable_grad():
            x_test = x.clone().detach()
            max_indices = torch.argmax(x_test, dim=-1, keepdim=True)
            predictions_one_hot = torch.zeros_like(x_test)
            predictions_one_hot = predictions_one_hot.scatter_(-1, max_indices, 1).requires_grad_(True)
            if self.use_constraint_graph:
                loss = self.loss_function(predictions_one_hot, binary_graph)
            else:
                loss = self.loss_function(predictions_one_hot)
            loss = loss.sum()
            gradients = torch.autograd.grad(loss, predictions_one_hot, retain_graph=False, create_graph=False, allow_unused=False)[0]
            scores = gradients.detach().norm(p=1, dim=-1)

        mean = scores.mean(dim=-1, keepdim=True)
        std = scores.std(dim=-1, keepdim=True)
        scores_norm = (scores - mean) / (std + eps)

        probs = torch.sigmoid(alpha * scores_norm)  # bias by score
        avg_probs = probs.mean(dim=-1, keepdim=True)  # average over variables
        scale = top_prob / (avg_probs + eps)
        probs_scaled = (probs * scale).clamp(max=1.0)
        mask = torch.rand_like(probs_scaled) < probs_scaled

        return variable_ind & mask


class GreedyConfidenceDestroyStrategy(DestroyStrategy):

    def __init__(self):
        super(GreedyConfidenceDestroyStrategy, self).__init__(name='greedyconfidence')

    def make_selection(self, threshold, inputs):
        """
        Greedy confidence-margin removal from logits.
        """
        variable_ind = inputs["variable_ind"]
        logits = inputs["logits"]

        top_prob = 1 - threshold

        # Determine how many variables to keep per instance based on the ratio.
        k_per, k_max = self.get_ks(top_prob, variable_ind)
        if k_max <= 0:
            return variable_ind

        confidence = F.softmax(logits, dim=-1)
        top2_vals, _ = torch.topk(confidence, k=2, dim=-1)  # shape [batch_size, var_count, 2]
        scores = top2_vals[..., 0] - top2_vals[..., 1]  # shape [batch_size, var_count]

        scores = scores.masked_fill(~variable_ind, float('-inf'))
        selected = self.get_selected(variable_ind, scores, k_per, k_max)

        return variable_ind & selected


class StochasticConfidenceDestroyStrategy(DestroyStrategy):

    def __init__(self):
        super(StochasticConfidenceDestroyStrategy, self).__init__(name='stochasticconfidence')

    def make_selection(self, threshold, inputs):
        """
        Stochastic confidence-margin removal from logits.
        """
        variable_ind = inputs["variable_ind"]
        logits = inputs["logits"]

        top_prob = 1 - threshold
        alpha = 1
        eps = 1e-8

        confidence = F.softmax(logits, dim=-1)
        top2_vals, _ = torch.topk(confidence, k=2, dim=-1)  # shape [batch_size, var_count, 2]
        scores = top2_vals[..., 0] - top2_vals[..., 1]  # shape [batch_size, var_count]

        mean = scores.mean(dim=-1, keepdim=True)
        std = scores.std(dim=-1, keepdim=True)
        scores_norm = (scores - mean) / (std + eps)

        # we use -alpha here since we want the lowest scores to have highest probs
        probs = torch.sigmoid(-alpha * scores_norm)  # bias by score
        avg_probs = probs.mean(dim=-1, keepdim=True)  # average over variables
        scale = top_prob / (avg_probs + eps)
        probs_scaled = (probs * scale).clamp(max=1.0)
        mask = torch.rand_like(probs_scaled) < probs_scaled

        return variable_ind & mask




class StochasticRelatedDestroyStrategy(DestroyStrategy):

    def __init__(self, use_constraint_graph=False, name='stochasticrelated'):
        super(StochasticRelatedDestroyStrategy, self).__init__(name=name)
        self.use_constraint_graph = use_constraint_graph
        if not self.use_constraint_graph:
            self.sudoku_M = self.make_sudoku_incidence()

    def make_sudoku_incidence(self):
        rows = torch.arange(9)
        cols = torch.arange(9)

        rr = rows.repeat_interleave(9)
        cc = cols.repeat(9)
        bb = (rr // 3) * 3 + (cc // 3)

        M_row = F.one_hot(rr, num_classes=9).T
        M_col = F.one_hot(cc, num_classes=9).T
        M_box = F.one_hot(bb, num_classes=9).T

        M = torch.cat([M_row, M_col, M_box], dim=0)  # 27 × 81
        return M.to(torch.float32)

    def make_selection(self, threshold, inputs):
        """
        Stochastic related removal based on sampled constraints.
        """

        variable_ind = inputs["variable_ind"]

        if not self.use_constraint_graph:
            # hard code for sudoku
            noise = torch.rand(variable_ind.size(0), 27, device=variable_ind.device)
            constraint_mask = (noise > threshold)  # Noise greater than threshold will remain "True" in mask
            variable_mask = (constraint_mask.to(torch.float32) @ self.sudoku_M.to(variable_ind.device)) > 0  # (B, 81) bool

        else:
            binary_graph = inputs["binary_constraint_graph"]
            constraint_graph = torch.tril(binary_graph, diagonal=-1)

            # Generate Uniform noise with the same shape as var_inds
            top_prob = 1 - threshold
            top_prob_scaled = top_prob * constraint_graph.size(-1) / constraint_graph.sum(dim=(1, 2))
            threshold_scaled = 1 - top_prob_scaled

            noise = torch.rand(constraint_graph.shape, device=variable_ind.device)
            constraint_mask = (noise > threshold_scaled.view(-1,1,1)) & constraint_graph # Noise greater than threshold will remain "True" in mask
            variable_mask = torch.any(constraint_mask, dim=-1) | torch.any(constraint_mask, dim=-2)

        # Update the var_inds by applying the mask
        return variable_ind & variable_mask


class GreedyRelatedDestroyStrategy(StochasticRelatedDestroyStrategy):

    def __init__(self, loss_function, use_constraint_graph):
        super(GreedyRelatedDestroyStrategy, self).__init__(use_constraint_graph, name="greedyrelated")
        self.loss_function = loss_function

    def make_selection(self, threshold, inputs):
        """
        Greedy related removal based on high-scoring constraints.
        """
        variable_ind = inputs["variable_ind"]
        x = inputs["soft_onehot"]
        batch_size, var_count, var_dim = x.size()

        if not self.use_constraint_graph:
            top_prob = 1.0 - threshold
            constraint_inds = torch.ones(batch_size, 27, dtype=torch.bool, device=variable_ind.device)

            k_per, k_max = self.get_ks(top_prob, constraint_inds)  # k_per[b] ≈ top_prob * 27
            if k_max <= 0:
                return variable_ind

            with torch.enable_grad():
                x_test = x.clone().detach()
                x_test.requires_grad_(True)

                max_indices = torch.argmax(x_test, dim=-1, keepdim=True)  # [B, 81, 1]
                predictions_one_hot = torch.zeros_like(x_test)
                predictions_one_hot = predictions_one_hot.scatter_(-1, max_indices, 1)
                predictions_one_hot.requires_grad_(True)

                loss = self.loss_function(predictions_one_hot)
                loss = loss.sum()
                gradients = torch.autograd.grad(loss, predictions_one_hot, retain_graph=False, create_graph=False, allow_unused=False)[0]

            var_scores = gradients.detach().norm(p=1, dim=-1)  # [B, 81]

            M = self.sudoku_M.to(variable_ind.device)
            constraint_scores = var_scores @ M.t()  # [B, 27]
            selected_constraints = self.get_selected(constraint_inds, constraint_scores, k_per, k_max)  # [B, 27] bool
            variable_mask = (selected_constraints.float() @ M) > 0  # [B, 81] bool

        else:
            binary_graph = inputs["binary_constraint_graph"]
            constraint_graph = torch.tril(binary_graph, diagonal=-1)  # don't double count the constraints

            top_prob = 1 - threshold
            constraint_count = constraint_graph.sum(dim=(1, 2))
            top_prob_scaled = top_prob * var_count / constraint_count
            constraint_inds = constraint_graph.view(batch_size, -1)

            # Determine how many variables to keep per instance based on the ratio.
            k_per, k_max = self.get_ks(top_prob_scaled, constraint_inds)
            if k_max <= 0:
                return variable_ind

            with torch.enable_grad():
                x_test = x.clone().detach()
                max_indices = torch.argmax(x_test, dim=-1, keepdim=True)
                predictions_one_hot = torch.zeros_like(x_test)
                predictions_one_hot = predictions_one_hot.scatter_(-1, max_indices, 1).requires_grad_(True)
                loss = self.loss_function(predictions_one_hot, binary_graph)
                loss = loss.sum()
                gradients = torch.autograd.grad(loss, predictions_one_hot, retain_graph=False, create_graph=False,
                                                allow_unused=False)[0]

            scores = gradients.detach().norm(p=1, dim=-1)  # [B, N]
            s_i = scores.unsqueeze(2)  # [B, N, 1]
            s_j = scores.unsqueeze(1)  # [B, 1, N]
            pair_scores = (s_i + s_j)  # [B, N, N]

            # Only care about actual constraints (lower triangle)
            pair_scores = pair_scores.masked_fill(~constraint_graph.bool(), float("-inf"))

            # Flatten to match constraint_inds
            constraint_scores = pair_scores.view(batch_size, -1)  # [B, M]

            # deterministic selection: take constraints with worst scores
            selected_flat = self.get_selected(
                constraint_inds,  # indicator: where real constraints exist
                constraint_scores,  # scores for each candidate constraint
                k_per, k_max
            )  # [B, M] bool

            selected_constraints = selected_flat.view(batch_size, var_count, var_count)  # [B, N, N] bool (lower-triangular)
            selected = selected_constraints.any(dim=-1) | selected_constraints.any(dim=-2)  # [B, N]

        return variable_ind & selected
