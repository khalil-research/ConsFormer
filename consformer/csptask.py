import torch

from abc import ABC, abstractmethod
from torch.utils.data import DataLoader, TensorDataset


class CSPTask(ABC):
    def __init__(self, loss_fn, acc_fn, task_name, has_labels):
        self.loss_fn = loss_fn
        self.acc_fn = acc_fn
        self.task_name = task_name
        self.has_labels = has_labels

    @abstractmethod
    def get_data_loaders(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_constraint_graph_adj_mat(self, *args, **kwargs):
        pass

    @abstractmethod
    def calculate_loss_and_accuracy(self, *args, **kwargs):
        pass


class SudokuTask(CSPTask):

    def __init__(self, loss_fn, acc_fn, task_name="sudoku", has_labels=True):
        super().__init__(loss_fn, acc_fn, task_name, has_labels)

    def process_inputs(self, x, y):
        """
        x, y has shape [10000, 9, 9, (One Hot Embed) 9]
          we want to convert them to [10000, (Embed_size) 9, (Seq_len) 81]
        is_input is a boolean tensor of shape [10000, 81],
          representing if the item in the sequence is a variable or not
        """
        is_input = x.sum(dim=-1, keepdim=True).int() == 0

        N, H, W, embed = x.size()
        x = x.view(N, H * W, embed)
        y = y.view(N, H * W, embed)
        is_input = is_input.view(N, H * W)

        constraint_graph = self.get_constraint_graph_adj_mat()
        constraint_graph = constraint_graph.repeat(N, 1, 1)

        return x, y, constraint_graph, is_input

    def get_data_loaders(self, input_path, labels_path, input_path_test, labels_path_test, batch_size):
        with open(input_path, 'rb') as f:
            x_in = torch.load(f, weights_only=True).to(torch.float32)
        with open(labels_path, 'rb') as f:
            y_in = torch.load(f, weights_only=True).to(torch.float32)
        with open(input_path_test, 'rb') as f:
            x_in_test = torch.load(f, weights_only=True).to(torch.float32)
        with open(labels_path_test, 'rb') as f:
            y_in_test = torch.load(f, weights_only=True).to(torch.float32)

        x, y, constraint_graph, is_input = self.process_inputs(x_in, y_in)
        sudoku_train = TensorDataset(x, y, constraint_graph, is_input)

        x_test, y_test, constraint_graph, is_input = self.process_inputs(x_in_test, y_in_test)
        sudoku_test = TensorDataset(x_test, y_test, constraint_graph, is_input)

        train_loader = DataLoader(sudoku_train, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(sudoku_test, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader

    def calculate_loss_and_accuracy(self, labels, predictions, constraint_graphs, var_inds, predictions_all=None):
        loss = self.loss_fn(predictions)
        all_diff_acc, total_predictions = self.acc_fn.get_accuracy(predictions)
        if predictions_all is not None:
            correct_cell_predictions, instances_solved, _ = self.acc_fn.get_accuracy_exact(predictions_all, labels, var_inds)
        else:
            correct_cell_predictions, instances_solved, _ = self.acc_fn.get_accuracy_exact(predictions, labels, var_inds)

        total_cell_predictions = torch.sum(var_inds).item()  # number of variables predicted

        return loss, {
            "all_diff": (all_diff_acc, total_predictions),
            "accuracy": (correct_cell_predictions, total_cell_predictions),
            "instance_accuracy": (instances_solved, total_predictions),
            # note, the accuracy reported for sudoku in the paper is the instance accuracy
        }

    def get_constraint_graph_adj_mat(self):
        N = 81
        adj_mat = torch.zeros((N, N), dtype=torch.bool)

        def get_row_col(idx):
            return idx // 9, idx % 9

        def get_grid(row, col):
            return (row // 3) * 3 + (col // 3)

        for i in range(N):
            for j in range(N):
                row_i, col_i = get_row_col(i)
                row_j, col_j = get_row_col(j)
                same_row = row_i == row_j
                same_col = col_i == col_j
                same_grid = get_grid(row_i, col_i) == get_grid(row_j, col_j)
                adj_mat[i, j] = same_row or same_col or same_grid

        return adj_mat


class GraphColoringTask(CSPTask):
    def __init__(self, loss_fn, acc_fn, task_name="graph_coloring", has_labels=False):
        super().__init__(loss_fn, acc_fn, task_name, has_labels)

    def get_data_loaders(self, input_path_train, input_path_test, batch_size, vertices_count, colors_count):
        graph_adj_mat_train = torch.load(input_path_train, weights_only=True)
        inputs_train = torch.zeros(graph_adj_mat_train.size(0), vertices_count, colors_count)
        var_inds_train = torch.ones(graph_adj_mat_train.size(0), vertices_count, dtype=torch.bool)
        constraint_graphs = self.get_constraint_graph_adj_mat(graph_adj_mat_train)
        train_dataset = TensorDataset(inputs_train, constraint_graphs, var_inds_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        graph_adj_mat_test = torch.load(input_path_test, weights_only=True)
        inputs_test = torch.zeros(graph_adj_mat_test.size(0), vertices_count, colors_count)
        var_inds_test = torch.ones(graph_adj_mat_test.size(0), vertices_count, dtype=torch.bool)
        constraint_graphs = self.get_constraint_graph_adj_mat(graph_adj_mat_test)
        test_dataset = TensorDataset(inputs_test, constraint_graphs, var_inds_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        return train_loader, test_loader

    def get_constraint_graph_adj_mat(self, graph_adj_mat):
        batch_size, seq_len, _ = graph_adj_mat.size()
        return graph_adj_mat.bool() + torch.eye(seq_len, seq_len, dtype=torch.bool).repeat(batch_size, 1, 1)

    def calculate_loss_and_accuracy(self, predictions, constraint_graph, var_inds):
        batch_size, seq_len, _ = constraint_graph.size()
        graph_adj_mat_without_self = constraint_graph.int() - torch.eye(seq_len, seq_len).repeat(batch_size, 1, 1)

        loss = self.loss_fn(predictions, graph_adj_mat_without_self)
        (num_violated, percent_violated, instances_solved), total_predictions = self.acc_fn.get_accuracy(predictions, graph_adj_mat_without_self)

        return loss, {
            "binary_inequality": (num_violated, total_predictions),
            "accuracy": (percent_violated, total_predictions),
            "instance_accuracy": (instances_solved, total_predictions),
            # note, the accuracy reported in the paper is the instance accuracy
        }


class NurseSchedulingTask(CSPTask):

    def __init__(self, loss_fn, acc_fn, task_name="nurse_scheduling", has_labels=False):
        super().__init__(loss_fn, acc_fn, task_name, has_labels)

    def get_data_loaders(self, input_path_train, input_path_test, batch_size, num_days, shifts_per_day, nurses_per_shift, nurses_count):

        with open(input_path_train, 'rb') as f:
            x_in = torch.load(f, weights_only=True).float()
        x_in = x_in.view(x_in.shape[0], -1, nurses_count)
        var_ind = x_in.sum(dim=-1, keepdim=True).int() == 0
        var_ind = var_ind.squeeze()
        constraint_graph = self.get_constraint_graph_adj_mat(num_days, shifts_per_day, nurses_per_shift)
        constraint_graph = constraint_graph.repeat(x_in.shape[0], 1, 1)
        dataset = TensorDataset(x_in, constraint_graph, var_ind)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        with open(input_path_test, 'rb') as f:
            x_in = torch.load(f, weights_only=True).float()
        x_in = x_in.view(x_in.shape[0], -1, nurses_count)
        var_ind = x_in.sum(dim=-1, keepdim=True).int() == 0
        var_ind = var_ind.squeeze()
        constraint_graph = self.get_constraint_graph_adj_mat(num_days, shifts_per_day, nurses_per_shift)
        constraint_graph = constraint_graph.repeat(x_in.shape[0], 1, 1)
        dataset = TensorDataset(x_in, constraint_graph, var_ind)
        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader

    def get_constraint_graph_adj_mat(self, num_days, shifts_per_day, nurses_per_shift):

        num_variables = num_days * shifts_per_day * nurses_per_shift
        adj_matrix = torch.zeros((num_variables, num_variables))

        # Map (day, shift, nurse_slot) to variable index
        def variable_to_index(day, shift, nurse_slot):
            return day * (shifts_per_day * nurses_per_shift) + shift * nurses_per_shift + nurse_slot

        # Daily uniqueness constraints
        for day in range(num_days):
            for shift1 in range(shifts_per_day):
                for slot1 in range(nurses_per_shift):
                    for shift2 in range(shifts_per_day):
                        for slot2 in range(nurses_per_shift):
                            if shift1 != shift2 or slot1 != slot2:
                                i = variable_to_index(day, shift1, slot1)
                                j = variable_to_index(day, shift2, slot2)
                                adj_matrix[i, j] = 1

        # Shift transition constraints
        for day in range(num_days - 1):
            for slot1 in range(nurses_per_shift):
                for slot2 in range(nurses_per_shift):
                    last_shift_idx = variable_to_index(day, shifts_per_day - 1, slot1)
                    first_shift_next_day_idx = variable_to_index(day + 1, 0, slot2)
                    adj_matrix[last_shift_idx, first_shift_next_day_idx] = 1
                    adj_matrix[first_shift_next_day_idx, last_shift_idx] = 1  # Undirected

        return adj_matrix

    def calculate_loss_and_accuracy(self, predictions, constraint_graph, var_inds):

        loss = self.loss_fn(predictions)
        acc, sat_alldiff, sat_ineq, instances_solved_batch, instances_solved, total_predictions = self.acc_fn.get_accuracy(predictions)

        return loss, {
            "sat_alldiff": (sat_alldiff, total_predictions),
            "sat_ineq": (sat_ineq, total_predictions),
            "accuracy": (acc, total_predictions),
            "instance_accuracy": (torch.sum(instances_solved), total_predictions),
            # note, the accuracy reported in the paper is the instance accuracy
        }


class MaxCutTask(CSPTask):
    def __init__(self, loss_fn, acc_fn, task_name="max_cut", has_labels=False):
        super().__init__(loss_fn, acc_fn, task_name, has_labels)

    def get_data_loaders(self, input_path_train, input_path_test, batch_size, vertices_count):
        graph_adj_mat_train = torch.load(input_path_train, weights_only=True)
        inputs_train = torch.zeros(graph_adj_mat_train.size(0), vertices_count, 2)
        var_inds_train = torch.ones(graph_adj_mat_train.size(0), vertices_count, dtype=torch.bool)
        constraint_graphs = self.get_constraint_graph_adj_mat(graph_adj_mat_train)
        train_dataset = TensorDataset(inputs_train, constraint_graphs, var_inds_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        graph_adj_mat_test = torch.load(input_path_test, weights_only=True)
        inputs_test = torch.zeros(graph_adj_mat_test.size(0), vertices_count, 2)
        var_inds_test = torch.ones(graph_adj_mat_test.size(0), vertices_count, dtype=torch.bool)
        constraint_graphs = self.get_constraint_graph_adj_mat(graph_adj_mat_test)
        test_dataset = TensorDataset(inputs_test, constraint_graphs, var_inds_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        return train_loader, test_loader

    def get_constraint_graph_adj_mat(self, graph_adj_mat):
        batch_size, seq_len, _ = graph_adj_mat.size()
        return graph_adj_mat.bool() + torch.eye(seq_len, seq_len, dtype=torch.bool).repeat(batch_size, 1, 1)

    def calculate_loss_and_accuracy(self, predictions, constraint_graph, var_inds):
        batch_size, seq_len, _ = constraint_graph.size()
        graph_adj_mat_without_self = constraint_graph.int() - torch.eye(seq_len, seq_len).repeat(batch_size, 1, 1)

        loss = self.loss_fn(predictions, graph_adj_mat_without_self)
        cut_sizes = self.acc_fn.get_per_instance_cut_size(predictions, graph_adj_mat_without_self)

        return loss, {
            "accuracy": (torch.sum(cut_sizes), batch_size),
        }


