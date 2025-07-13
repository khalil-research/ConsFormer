import torch
import torch.nn as nn
import time

from torch import optim
from torch import Tensor

from .csptask import CSPTask



class Trainer:
    def __init__(
        self,
        model: nn.Module,
        csp_task: CSPTask,
        device: torch.device,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader = None,
        optimizer: optim.Optimizer = None,
        scheduler = None,
        num_epochs: int = 1000,
        learning_rate: float = 0.001,
        log_interval: int = 10,
        model_name: str = "model",
    ):
        """
        Initializes the Trainer object with all necessary components.
        """
        self.device = device
        self.model = model.to(self.device)
        self.csp_task = csp_task
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_epochs = num_epochs
        self.log_interval = log_interval
        self.model_name = model_name

        # Optimizer and scheduler
        self.optimizer = optimizer if optimizer else optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = scheduler

        # Loss function from training strategy
        self.loss_fn = self.csp_task.loss_fn


    def _get_var_indices(self, x:Tensor):
        """
        x has size (batch_size, seq_len, embedding_size)
        we want to mask of size (batch_size, seq_len) representing the items need to be masked (ie the variables not yet assigned)
        in the tensor, they are represented by a embedding consisting only of zeros
        """
        mask = torch.sum(x, dim=-1) == 0  # check if none of the elements is 1 (i.e., sum is 0)
        return mask

    def _run_epoch_iter(self, loader, phase: str, iters=1):
        """
        Runs a single epoch with iterative calling specified by iter for the given phase: 'training', 'validation', or 'test'.
        """
        is_train = phase == 'training'
        self.model.train() if is_train else self.model.eval()

        total_loss = 0.0
        num_batches = 0
        epoch_stats: dict[str, dict[str, float]] = {}

        with torch.set_grad_enabled(is_train):
            for data_instance in loader:
                if self.csp_task.has_labels:
                    inputs, labels, constraint_graphs, var_inds = data_instance
                    labels = labels.to(self.device)
                else:
                    inputs, constraint_graphs, var_inds = data_instance
                inputs = inputs.to(self.device)
                constraint_graphs = constraint_graphs.to(self.device)
                var_inds = var_inds.to(self.device)

                #  if there are unfilled variables, intialize randomly
                unfilled_inds = self._get_var_indices(inputs)
                if torch.any(unfilled_inds):
                  random_assign = torch.softmax(torch.rand(inputs[unfilled_inds].size()), dim=-1)
                  inputs[unfilled_inds] = random_assign.to(self.device)

                # Generate predictions in a loop
                inputs_iter = inputs
                for iter_step in range(iters):
                    generator_preds_onehot, out_logits = self.model(inputs_iter, constraint_graphs, var_inds)
                    inputs_iter = generator_preds_onehot

                # Loss and metrics computation
                if is_train:
                    if self.csp_task.has_labels:
                        loss, batch_metrics = self.csp_task.calculate_loss_and_accuracy(labels, generator_preds_onehot, constraint_graphs, var_inds)
                    else:
                        loss, batch_metrics = self.csp_task.calculate_loss_and_accuracy(generator_preds_onehot, constraint_graphs, var_inds)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    if self.scheduler:
                        self.scheduler.step()

                else:
                    if self.csp_task.has_labels:
                        loss, batch_metrics = self.csp_task.calculate_loss_and_accuracy(labels, generator_preds_onehot, constraint_graphs, var_inds)
                    else:
                        loss, batch_metrics = self.csp_task.calculate_loss_and_accuracy(generator_preds_onehot, constraint_graphs, var_inds)

                total_loss += loss.item()
                num_batches += 1

                # update every metric accumulator
                for name, (correct, total) in batch_metrics.items():
                    stats = epoch_stats.setdefault(name, {"correct": 0.0, "total": 0.0})
                    stats["correct"] += correct.item()  # or .sum()
                    stats["total"] += total

        avg_loss = total_loss / num_batches

        # compute final accuracies
        epoch_results = {
            name: stats["correct"] / stats["total"]
            for name, stats in epoch_stats.items()
        }

        # print summary
        metrics_str = " ".join(f"{k}={v:.4f}" for k, v in epoch_results.items())
        print(f"{phase.title()} loss={avg_loss:.4f} {metrics_str}")

        return avg_loss, epoch_results["accuracy"]

    def train(self):
        """
        Runs the full training loop
        """
        best_test_acc = 0
        training_start_time = time.time()
        for epoch in range(1, self.num_epochs + 1):
            print(f"\nEpoch {epoch}/{self.num_epochs}")
            train_loss, train_accuracy= self._run_epoch_iter(self.train_loader, 'training', 1)

            if self.scheduler:
                self.scheduler.step()

            if self.test_loader and (epoch % self.log_interval == 0 or epoch == self.num_epochs):
                test_loss, test_accuracy = self._run_epoch_iter(self.test_loader, 'test', 10)
                if test_accuracy > best_test_acc:
                    best_test_acc = test_accuracy
                    torch.save(self.model.state_dict(), "./saved_models/" + self.model_name + f"_best")

                    current_time = time.time() - training_start_time
                    print(f"Best model saved in {current_time} seconds at epoch {epoch}")

        torch.save(self.model.state_dict(), "./saved_models/" + self.model_name + "_final")
        print("\nTraining complete.")

    def evaluate_by_iters(self, loader, phase: str = 'test', iters=[1]):
        """
        Run the model on a batch of instances iteratively until a maximum iteration is reached.
        """
        self.model.eval()
        max_iters = max(iters)
        epoch_accuracy = {}
        time_elapsed = {}
        total_predictions = 0  # how many instances are there in total

        for iterations in iters:
            epoch_accuracy[iterations] = 0
            time_elapsed[iterations] = 0

        with torch.set_grad_enabled(False):

            for data_instance in loader:
                start_time = time.time()
                if self.csp_task.has_labels:
                    inputs, labels, constraint_graphs, var_inds = data_instance
                    labels = labels.to(self.device)
                else:
                    inputs, constraint_graphs, var_inds = data_instance
                inputs = inputs.to(self.device)
                constraint_graphs = constraint_graphs.to(self.device)
                var_inds = var_inds.to(self.device)

                batch_size, seq_len, embedding_size = inputs.size()

                # if there are unfilled variables, initialize randomly
                unfilled_inds = self._get_var_indices(inputs)
                if torch.any(unfilled_inds):
                    random_assign = torch.softmax(torch.rand(inputs[unfilled_inds].size()), dim=-1)
                    inputs[unfilled_inds] = random_assign.to(self.device)

                solved_instances = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

                inputs_iter = inputs
                for iter_step in range(1, max_iters + 1):
                    generator_preds_onehot, out_logits = self.model(inputs_iter, constraint_graphs, var_inds)
                    correct_instances = self.csp_task.acc_fn.get_per_instance_accuracy(generator_preds_onehot)
                    solved_instances |= correct_instances
                    inputs_iter = generator_preds_onehot

                    if iter_step in iters:
                        end_time = time.time()
                        correct_instances_count = torch.sum(solved_instances)
                        epoch_accuracy[iter_step] += correct_instances_count.item()
                        time_elapsed[iter_step] += end_time - start_time

                total_predictions += batch_size

        for iterations in iters:
            epoch_accuracy[iterations] /= total_predictions

            print(f"{phase.capitalize()} {iterations} Iterations, Accuracy: {epoch_accuracy[iterations]:.4f}, Time Elapsed: {time_elapsed[iterations]}")

        return epoch_accuracy, time_elapsed

    def evaluate_by_time_single_instance(self, loader, phase: str = 'test', time_limits=[1]):
        """
        Run the model on a single instance iteratively until a time limit is reached.
        """

        is_train = False
        self.model.eval()

        max_iters = 100000

        epoch_accuracy = {}
        time_elapsed = {}
        iters_used = {}

        for timestamp in time_limits:
            epoch_accuracy[timestamp] = 0
            time_elapsed[timestamp] = 0
            iters_used[timestamp] = 0
        total_predictions = 0  # how many instances are there in total

        with torch.set_grad_enabled(is_train):
            for data_instance in loader:
                start_time = time.time()
                if self.csp_task.has_labels:
                    inputs, labels, constraint_graphs, var_inds = data_instance
                    labels = labels.to(self.device)
                else:
                    inputs, constraint_graphs, var_inds = data_instance
                inputs = inputs.to(self.device)
                constraint_graphs = constraint_graphs.to(self.device)
                var_inds = var_inds.to(self.device)

                batch_size, var_seq_len, embed_size = inputs.size()

                # if there are unfilled variables, intialize randomly
                unfilled_inds = self._get_var_indices(inputs)
                if torch.any(unfilled_inds):
                    random_assign = torch.softmax(torch.rand(inputs[unfilled_inds].size()), dim=-1)
                    inputs[unfilled_inds] = random_assign.to(self.device)

                solved_instances = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
                inputs_iter = inputs
                curr_time_idx = 0
                for iter_step in range(1, max_iters+1):
                    generator_preds_onehot, out_logits = self.model(inputs_iter, constraint_graphs, var_inds)
                    inputs_iter = generator_preds_onehot
                    correct_instances = self.csp_task.acc_fn.get_per_instance_accuracy(generator_preds_onehot, constraint_graphs, var_inds)
                    solved_instances |= correct_instances

                    time_spent = time.time() - start_time
                    if time_spent >= time_limits[curr_time_idx]:

                        time_elapsed[time_limits[curr_time_idx]] += time_spent
                        iters_used[time_limits[curr_time_idx]] += iter_step

                        # if any of the pool of candidates are solved, the instance is solved
                        if torch.any(solved_instances):
                            epoch_accuracy[time_limits[curr_time_idx]] += batch_size

                        else:
                            epoch_accuracy[time_limits[curr_time_idx]] += torch.sum(solved_instances).item()

                        curr_time_idx += 1
                        if curr_time_idx == len(time_limits):
                            break

                total_predictions += batch_size

        for timestamp in time_limits:
            epoch_accuracy[timestamp] /= total_predictions  # how many instances were completely correct

        return epoch_accuracy, time_elapsed, iters_used


class MaxCutTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def evaluate_by_time_single_instance(self, loader, phase: str = 'test', time_limits=[1]):

        is_train = False
        self.model.eval()

        max_iters = 100000

        epoch_cut = {}
        time_elapsed = {}
        iters_used = {}

        for timestamp in time_limits:
            epoch_cut[timestamp] = 0
            time_elapsed[timestamp] = 0
            iters_used[timestamp] = 0
        total_predictions = 0  # how many instances are there in total
        best_cut_val = 0

        with torch.set_grad_enabled(is_train):
            for data_instance in loader:
                start_time = time.time()
                if self.csp_task.has_labels:
                    inputs, labels, constraint_graphs, var_inds = data_instance
                    labels = labels.to(self.device)
                else:
                    inputs, constraint_graphs, var_inds = data_instance
                inputs = inputs.to(self.device)
                constraint_graphs = constraint_graphs.to(self.device)
                var_inds = var_inds.to(self.device)

                batch_size, var_seq_len, embed_size = inputs.size()

                # if there are unfilled variables, intialize randomly
                unfilled_inds = self._get_var_indices(inputs)
                if torch.any(unfilled_inds):
                    random_assign = torch.softmax(torch.rand(inputs[unfilled_inds].size()), dim=-1)
                    inputs[unfilled_inds] = random_assign.to(self.device)

                current_best_cut = torch.zeros(batch_size, device=self.device)
                inputs_iter = inputs
                curr_time_idx = 0
                for iter_step in range(1, max_iters + 1):
                    generator_preds_onehot, out_logits = self.model(inputs_iter, constraint_graphs, var_inds)
                    inputs_iter = generator_preds_onehot

                    cut_sizes = self.csp_task.acc_fn.get_per_instance_cut_size(generator_preds_onehot,
                                                                               constraint_graphs)
                    curr_cut, curr_index = torch.max(cut_sizes, dim=0)
                    if curr_cut > best_cut_val:
                        ret_sol = generator_preds_onehot[curr_index]

                    current_best_cut = torch.maximum(current_best_cut, cut_sizes)

                    time_spent = time.time() - start_time
                    if time_spent >= time_limits[curr_time_idx]:

                        time_elapsed[time_limits[curr_time_idx]] += time_spent
                        iters_used[time_limits[curr_time_idx]] += iter_step
                        epoch_cut[time_limits[curr_time_idx]] = torch.max(current_best_cut, dim=0)[0].item()

                        curr_time_idx += 1
                        if curr_time_idx == len(time_limits):
                            break

                total_predictions += batch_size

        return epoch_cut, time_elapsed, iters_used
