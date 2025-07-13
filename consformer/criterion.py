import torch
import torch.nn.functional as F
from torch import nn
from abc import abstractmethod


class CustomSudokuAccuracy():
    def __init__(self):
        super(CustomSudokuAccuracy, self).__init__()

    # Helper function to extract rows, columns, and 3x3 subgrids
    def extract_sudoku_segments(self, x):
        """
        takes in one hot predictions of the model, size (batch_size, seq_len (81), embed_size (9))
        returns rows, columns, and 3x3 subgrids of the input each of size
        (batch_size, sub_seq_count (9 rows/cols/grids), sub_seq_len (9 in each row/col/grid), embed_size(onehot 9))
        to calculate the loss for each batch
        """

        batch_size, seq_len, embed_size = x.size()
        rows = x.view(batch_size, 9, 9, embed_size)
        columns = rows.permute(0, 2, 1, 3)  # (batch_size, 9, 9, embed_size)

        grids = []
        for i in range(3):
            for j in range(3):
                grid = rows[:, i*3:(i+1)*3, j*3:(j+1)*3, :]  # Shape: (batch_size, 3, 3, embed_size)
                grids.append(grid)
        grids = torch.stack(grids, dim=1)  # (batch_size, 9, 3, 3, embed_size)
        grids = grids.view(batch_size, 9, 9, embed_size)

        return rows, columns, grids

    def _batch_acc(self, x):
        """
        x has size (batch_size*sub_seq_count, seq_len (9), embed_size (9))
        calculate the accuracy of x where the accuracy is defined as
        the count of how many all diff is satisfied (ie all 9 values must all be 1)
        there are 27 all diff constraints per sudoku instance
        """

        max_indices = torch.argmax(x, dim=-1, keepdim=True)
        predictions_one_hot = torch.zeros_like(x)
        predictions_one_hot.scatter_(-1, max_indices, 1)
        x = predictions_one_hot

        x_sum_seq = x.sum(dim=1)
        target = torch.ones_like(x_sum_seq)

        acc_loose = torch.sum(x_sum_seq == target)

        acc = torch.all(x_sum_seq == target, dim=-1).float()
        acc = torch.sum(acc)

        return (acc, acc_loose)

    def get_accuracy(self, predictions):
        """
        computes the accuracy based on the evaluation of the constraints
        output has size (batch_size, seq_len (81), embed_size (9))
        """

        rows, columns, grids = self.extract_sudoku_segments(predictions)

        batch_size, sub_seq_count, sub_seq_len, embed_size = rows.size()
        combined_size = batch_size * sub_seq_count

        accs = [self._batch_acc(rows.reshape(combined_size, sub_seq_len, embed_size)),
                self._batch_acc(columns.reshape(combined_size, sub_seq_len, embed_size)),
                self._batch_acc(grids.reshape(combined_size, sub_seq_len, embed_size))]

        acc, acc_loose = 0, 0
        for a in accs:
            acc += a[0]
            acc_loose += a[1]

        # acc is the # of correct all-diff (max is 27),
        return acc, batch_size

    def get_accuracy_exact(self, predictions, labels, masks):
        """
        if labels are available for the task, use this function to get the exact accuracy using the labels
        output has size (batch_size, seq_len (81), embed_size (9))
        """
        batch_size, _, _ = predictions.size()

        max_indices = torch.argmax(predictions, dim=-1, keepdim=True)
        predictions_one_hot = torch.zeros_like(predictions)
        predictions_one_hot.scatter_(-1, max_indices, 1)
        predictions = predictions_one_hot

        labels_batch_size, _, _ = labels.size()

        correct = predictions == labels
        acc_ex = torch.sum(torch.all(correct[masks], dim=-1))
        acc_ex_whole = torch.sum(torch.all(correct.view(batch_size, -1), dim=-1))

        return acc_ex, acc_ex_whole, labels_batch_size


    def _batch_acc_per_instance(self, x):
        """
        x has size (batch_size*sub_seq_count, seq_len (9), embed_size (9))
        calculate the accuracy of x
        """
        # x = torch.round(x)

        x_sum_seq = x.sum(dim=1)  # batch_size*sub_seq_count, seq_len
        target = torch.ones_like(x_sum_seq)

        acc = torch.all(x_sum_seq == target, dim=-1).float()
        return acc

    def get_per_instance_accuracy(self, predictions):
        """
        this returns a binary tensor that indicates if each prediction in the batch is correct or not
        predictions has size (batch_size, seq_len (81), embed_size (9))
        """

        max_indices = torch.argmax(predictions, dim=-1, keepdim=True)
        predictions_one_hot = torch.zeros_like(predictions)
        predictions_one_hot.scatter_(-1, max_indices, 1)
        predictions = predictions_one_hot

        rows, columns, grids = self.extract_sudoku_segments(predictions)

        batch_size, sub_seq_count, sub_seq_len, embed_size = rows.size()
        combined_size = batch_size * sub_seq_count

        row_acc = self._batch_acc_per_instance(rows.reshape(combined_size, sub_seq_len, embed_size)).view(batch_size, sub_seq_count)
        col_acc = self._batch_acc_per_instance(columns.reshape(combined_size, sub_seq_len, embed_size)).view(batch_size, sub_seq_count)
        box_acc = self._batch_acc_per_instance(grids.reshape(combined_size, sub_seq_len, embed_size)).view(batch_size, sub_seq_count)

        acc = torch.concat((row_acc, col_acc, box_acc), dim=-1)  # batch_size, 27
        instance_correct = torch.all(acc, dim=-1)

        return instance_correct


class CustomGCOLAccuracy():
    def __init__(self):
        super(CustomGCOLAccuracy, self).__init__()

    def get_accuracy(self, predictions, adj_mat, predictions_all=None):
        """
        x has size (batch_size, seq_len (num_vertices), embed_size (num_colors))
        calculate the performance metrics of x
        """

        batch_size, _, _ = predictions.size()

        # ============= one-hot-ify prediction ===========
        max_indices = torch.argmax(predictions, dim=-1, keepdim=True)
        predictions_one_hot = torch.zeros_like(predictions, dtype=torch.float32)
        predictions_one_hot.scatter_(-1, max_indices, 1)

        # ============= feasibility ======================
        # Compute dot product between all node pairs, only nodes with the same color assignement will have value 1
        dot_products = torch.bmm(predictions_one_hot, predictions_one_hot.transpose(1, 2))  # (batch, vertex_count, vertex_count)
        # Mask the dot products with the adjacency matrix, only adjacent nodes with the same color will have value 1 (ie, violated constraints)
        num_violated = torch.sum(dot_products * adj_mat, dim=(1,2))  # Only keep values for edges
        percent_violated = torch.sum(num_violated / (adj_mat.sum(dim=(1,2)) + 1e-6))

        instances_solved = torch.sum(num_violated == 0)

        num_violated_total = torch.sum(num_violated)

        return (num_violated_total, percent_violated, instances_solved), batch_size

    def get_per_instance_accuracy(self, predictions, adj_mat, var_inds):

        # ============= one-hot-ify prediction ===========
        max_indices = torch.argmax(predictions, dim=-1, keepdim=True)
        predictions_one_hot = torch.zeros_like(predictions, dtype=torch.float32)
        predictions_one_hot.scatter_(-1, max_indices, 1)

        # ============= feasibility ======================
        # Compute dot product between all node pairs, only nodes with the same color assignement will have value 1
        dot_products = torch.bmm(predictions_one_hot, predictions_one_hot.transpose(1, 2))  # (batch, vertex_count, vertex_count)
        # Mask the dot products with the adjacency matrix, only adjacent nodes with the same color will have value 1 (ie, violated constraints)
        num_violated = torch.sum(dot_products * adj_mat, dim=(1,2))  # Only keep values for edges
        instances_solved = num_violated == 0

        return instances_solved


class CustomCUTAccuracy():
    def __init__(self):
        super(CustomCUTAccuracy, self).__init__()


    def get_accuracy(self, predictions, adj_mat, labels):
        """
        x has size (batch_size, seq_len (num_vertices), embed_size (num_colors))
        calculate the performance metrics of x
        """

        batch_size, _, _ = predictions.size()

        # ============= one-hot-ify prediction ===========
        max_indices = torch.argmax(predictions, dim=-1, keepdim=True)
        predictions_one_hot = torch.zeros_like(predictions, dtype=torch.float32)
        predictions_one_hot.scatter_(-1, max_indices, 1)

        # ============= best (maximum) cut size found ======================
        dot_products = 1 - torch.bmm(predictions_one_hot, predictions_one_hot.transpose(1, 2))  # (batch, vertex_count, vertex_count)
        cut_size = torch.sum(dot_products * adj_mat, dim=(1,2))  # sum over all edges
        gap = cut_size - labels

        gap_total = torch.mean(gap)
        percent_beat = torch.sum(gap < 0) / batch_size

        return (gap_total, percent_beat), batch_size

    def get_per_instance_cut_size(self, predictions, adj_mat):

        # ============= one-hot-ify prediction ===========
        max_indices = torch.argmax(predictions, dim=-1, keepdim=True)
        predictions_one_hot = torch.zeros_like(predictions, dtype=torch.float32)
        predictions_one_hot.scatter_(-1, max_indices, 1)

        # ============= best (maximum) cut size found ======================
        dot_products = 1 - torch.bmm(predictions_one_hot, predictions_one_hot.transpose(1, 2))  # (batch, vertex_count, vertex_count)
        cut_size = torch.sum(dot_products * adj_mat, dim=(1, 2))  # sum over all edges

        return cut_size // 2


class CustomNRAccuracy():
    def __init__(self, hard=True):
        super(CustomNRAccuracy, self).__init__()
        self.hard = hard

    def get_accuracy(self, predictions):

        batch_size, _, num_nurses = predictions.size()
        predictions = predictions.view(batch_size, -1, 3, 3, num_nurses)
        batch_size, num_days, shifts_per_day, nurses_per_shift, num_nurses = predictions.size()

        max_indices = torch.argmax(predictions, dim=-1, keepdim=True)
        predictions_one_hot = torch.zeros_like(predictions)
        predictions_one_hot.scatter_(-1, max_indices, 1)
        predictions = predictions_one_hot

        # constraint 1: day all diff
        predictions = predictions.view(batch_size*num_days, -1, num_nurses)
        day_sum_nurse = predictions.sum(dim=1)  # (batch_size*self.num_days, num_nurses)
        target = torch.ones_like(day_sum_nurse)

        day_alldiff = day_sum_nurse <= 1  # (batch_size*self.num_days, num_nurses)
        day_alldiff = day_alldiff.view(batch_size, num_days, num_nurses)
        # day_alldiff_card = torch.sum(day_alldiff, dim=-1)  # (batch_size, num_days)
        day_alldiff_solved_per = torch.all(day_alldiff, dim=-1)
        day_alldiff_solved_instance = torch.all(day_alldiff, dim=(-2,-1))

        # constraint 2: inequality
        predictions = predictions.view(batch_size, num_days, shifts_per_day, nurses_per_shift, num_nurses)
        last_shift = predictions[:, :-1, -1, :, :].sum(dim=-2) #(batch_size, num_days-1, num_nurses)
        first_shift = predictions[:, 1:, 0, :, :].sum(dim=-2) #(batch_size, num_days-1, num_nurses)
        shift_overlap = last_shift * first_shift
        shift_overlap = shift_overlap.sum(dim=-1)  #(batch_size, num_days-1, num_nurses) -> (batch_size, num_days-1)
        shift_inequality = shift_overlap == 0
        shift_inequality_solved_per = torch.sum(shift_inequality, dim=-1)
        shift_inequality_solved_instance = torch.all(shift_inequality, dim=-1)

        day_alldiff_solved_per = torch.sum(day_alldiff_solved_per)  # /num_days to get percent
        shift_inequality_solved_per = torch.sum(shift_inequality_solved_per) # /num_days-1 to get percent
        instances_solved = shift_inequality_solved_instance & day_alldiff_solved_instance
        instances_solved_batch = torch.sum(instances_solved)
        acc = (day_alldiff_solved_per + shift_inequality_solved_per) / (num_days + num_days - 1)

        sat_alldiff = day_alldiff_solved_per
        sat_ineq = shift_inequality_solved_per

        return acc, sat_alldiff, sat_ineq, instances_solved_batch, instances_solved, batch_size

    def get_per_instance_accuracy(self, predictions):

        batch_size, _, num_nurses = predictions.size()
        predictions = predictions.view(batch_size, -1, 3, 3, num_nurses)
        batch_size, num_days, shifts_per_day, nurses_per_shift, num_nurses = predictions.size()

        max_indices = torch.argmax(predictions, dim=-1, keepdim=True)
        predictions_one_hot = torch.zeros_like(predictions)
        predictions_one_hot.scatter_(-1, max_indices, 1)
        predictions = predictions_one_hot

        # constraint 1: day all diff
        predictions = predictions.view(batch_size*num_days, -1, num_nurses)
        day_sum_nurse = predictions.sum(dim=1)  # (batch_size*self.num_days, num_nurses)

        day_alldiff = day_sum_nurse <= 1  # (batch_size*self.num_days, num_nurses)
        day_alldiff = day_alldiff.view(batch_size, num_days, num_nurses)
        day_alldiff_solved_instance = torch.all(day_alldiff, dim=(-2,-1))

        # constraint 2: inequality
        predictions = predictions.view(batch_size, num_days, shifts_per_day, nurses_per_shift, num_nurses)
        last_shift = predictions[:, :-1, -1, :, :].sum(dim=-2) #(batch_size, num_days-1, num_nurses)
        first_shift = predictions[:, 1:, 0, :, :].sum(dim=-2) #(batch_size, num_days-1, num_nurses)
        shift_overlap = last_shift * first_shift
        shift_overlap = shift_overlap.sum(dim=-1)  #(batch_size, num_days-1, num_nurses) -> (batch_size, num_days-1)
        shift_inequality = shift_overlap == 0
        shift_inequality_solved_instance = torch.all(shift_inequality, dim=-1)

        instances_solved = shift_inequality_solved_instance & day_alldiff_solved_instance

        return instances_solved


class CustomSudokuLoss(nn.Module):
    def __init__(self):
        super(CustomSudokuLoss, self).__init__()

    def extract_sudoku_segments(self, x):
        """
        takes in one hot predictions of the model, size (batch_size, seq_len (81), embed_size (9))
        returns rows, columns, and 3x3 subgrids of the input each of size
              (batch_size, sub_seq_count (9 rows/cols/grids), sub_seq_len (9 in each row/col/grid), embed_size(onehot 9))
        to calculate the loss for each batch, should use view and combine the first two dimensions
        """

        batch_size, seq_len, embed_size = x.size()

        rows = x.view(batch_size, 9, 9, embed_size)
        columns = rows.permute(0, 2, 1, 3)  # (batch_size, 9, 9, embed_size)

        grids = []
        for i in range(3):
            for j in range(3):
                # Extract the 3x3 block starting at (i*3, j*3)
                grid = rows[:, i*3:(i+1)*3, j*3:(j+1)*3, :]  # Shape: (batch_size, 3, 3, embed_size)
                grids.append(grid)

        grids = torch.stack(grids, dim=1)  # (batch_size, 9, 3, 3, embed_size)
        grids = grids.view(batch_size, 9, 9, embed_size)

        return rows, columns, grids

    @abstractmethod
    def batch_loss(self, x):
        """
        batch loss should take in tensor of size (n, seq_len, embed_size)
        and assume that each item in seq_len is all different
        """
        pass


    def forward(self, predictions):
        """
        output has size (batch_size, seq_len (81), embed_size (9))
        """

        rows, columns, grids = self.extract_sudoku_segments(predictions)

        # batch_size, how many rows/col/box, how many numbers in each row/col/box, dim of each number embedding
        batch_size, sub_seq_count, sub_seq_len, embed_size = rows.size()
        combined_size = batch_size * sub_seq_count

        loss = (self.batch_loss(rows.reshape(combined_size, sub_seq_len, embed_size)) +
                self.batch_loss(columns.reshape(combined_size, sub_seq_len, embed_size)) +
                self.batch_loss(grids.reshape(combined_size, sub_seq_len, embed_size)))

        # Average the loss over the batch
        return loss / batch_size


class CustomSudokuLossDecomposedMSE(CustomSudokuLoss):
    def __init__(self):
        super(CustomSudokuLossDecomposedMSE, self).__init__()
        self.mse_loss = nn.MSELoss()

    def batch_loss(self, x):
        """
        x has size (batch_size*sub_seq_count, seq_len (9), embed_size (9))
        calculate the loss of x
        """

        # Compute all pairwise dot products (n, 9, 9)
        dot_products = torch.bmm(x, x.transpose(1, 2))

        # Create a mask to select only upper-triangular entries (i < j), excluding diagonal
        mask = torch.triu(torch.ones(9, 9, dtype=torch.bool), diagonal=1)  # shape: (9, 9)

        # Apply the mask to each batch
        masked_dot_products = dot_products[:, mask]  # shape: (n, num_pairs), num_pairs = 36
        targets = torch.zeros_like(masked_dot_products)
        loss = self.mse_loss(masked_dot_products, targets)

        return loss


class CustomSudokuLossABSE(CustomSudokuLoss):
    def __init__(self, temp=1):
        super(CustomSudokuLossABSE, self).__init__()

        self.temp = temp

    def batch_loss(self, x):
        """
        x has size (batch_size*sub_seq_count, seq_len (9), embed_size (9))
        calculate the loss of x
        """

        # Sum over the seq_len dimension
        sum_seq_len = torch.sum(x, dim=1)  # Shape: (batch_size, embed_size)

        # Compute the squared difference from 1
        abs_diff = torch.abs(sum_seq_len - 1)  # Shape: (batch_size, embed_size)

        loss = torch.sum(abs_diff)  # Sum over the batch dimension

        return loss


class CustomSudokuLossMSE(CustomSudokuLoss):
    def __init__(self):
        super(CustomSudokuLossMSE, self).__init__()
        self.mse_loss = nn.MSELoss()

    def batch_loss(self, x):
        """
        x has size (batch_size*sub_seq_count, seq_len (9), embed_size (9))
        calculate the loss of x
        """
        x_sum_seq = x.sum(dim=1)
        target = torch.ones_like(x_sum_seq)  # we want each number in x_sum_eq to be one
        loss = self.mse_loss(x_sum_seq, target)

        return loss


class CustomGCOLLoss(nn.Module):
    def __init__(self):
        super(CustomGCOLLoss, self).__init__()

    @abstractmethod
    def constraint_approx(self, x, adj_matrix):
        pass


    def forward(self, predictions, adj_matrix):
        """
        predictions has size (batch_size, seq_len (vertex_count), embed_size (color_count))
        adj_matrix has size (batch_size, vertex_count, vertex_count)
        """
        loss = self.constraint_approx(predictions, adj_matrix)

        return loss


class CustomGCOLLossDot(CustomGCOLLoss):
    def __init__(self):
        super(CustomGCOLLossDot, self).__init__()

    def constraint_approx(self, x, adj_matrix):
        # for every pair of nodes identified by adj matrix, get their dot product

        # Compute dot product between all node pairs
        dot_products = torch.bmm(x, x.transpose(1, 2))  # (batch, vertex_count, vertex_count)

        # Mask the dot products with the adjacency matrix
        loss = torch.sum(dot_products * adj_matrix)  # Only keep values for edges

        return loss


class CustomGCOLLossDotMSE(CustomGCOLLoss):
    def __init__(self):
        super(CustomGCOLLossDotMSE, self).__init__()
        self.mse_loss = nn.MSELoss()

    def constraint_approx(self, x, adj_matrix):
        # for every pair of nodes identified by adj matrix, get their dot product

        # Compute dot product between all node pairs
        dot_products = torch.bmm(x, x.transpose(1, 2))  # (batch, vertex_count, vertex_count)
        approx_violated_constraints = dot_products * adj_matrix

        targets = torch.zeros_like(dot_products)

        loss = self.mse_loss(approx_violated_constraints, targets)

        return loss


class CustomGCOLLossDotEXP(CustomGCOLLoss):
    def __init__(self, temp=1):
        super(CustomGCOLLossDotEXP, self).__init__()
        self.temp = temp

    def constraint_approx(self, x, adj_matrix):
        """
        x has size (batch_size, seq_len (city_count), embed_size (city_count))
        x_ij is 1 if the city j is the ith city in the tour
        calculate the loss as an approximate of constraint violation (one single all diff constraint)
        """

        dot_products = torch.bmm(x, x.transpose(1, 2))  # (batch, vertex_count, vertex_count)
        approx_violated_constraints = dot_products * adj_matrix

        loss = torch.exp(self.temp * approx_violated_constraints)
        loss = torch.sum(loss)  # Sum over the batch dimension

        return loss


class CustomCUTLoss(nn.Module):
    def __init__(self):
        super(CustomCUTLoss, self).__init__()


    @abstractmethod
    def constraint_approx(self, x, adj_matrix):
        pass


    def forward(self, predictions, adj_matrix):
        """
        predictions has size (batch_size, seq_len (vertex_count), embed_size (color_count))
        adj_matrix has size (batch_size, vertex_count, vertex_count)
        """
        loss = self.constraint_approx(predictions, adj_matrix)

        return loss


class CustomCUTLossDot(CustomCUTLoss):
    def __init__(self):
        super(CustomCUTLossDot, self).__init__()

    def constraint_approx(self, x, adj_matrix):
        # for every pair of nodes identified by adj matrix, get their dot product

        # Compute dot product between all node pairs
        # dot product is 1 if the two nodes have the same assignment
        dot_products = torch.bmm(x, x.transpose(1, 2))  # (batch, vertex_count, vertex_count)

        # Mask the dot products with the adjacency matrix
        loss = torch.sum(dot_products * adj_matrix)  # Only keep values for edges

        return loss

class CustomCUTLossDotMSE(CustomGCOLLoss):
    def __init__(self):
        super(CustomCUTLossDotMSE, self).__init__()
        self.mse_loss = nn.MSELoss()

    def constraint_approx(self, x, adj_matrix):
        # for every pair of nodes identified by adj matrix, get their dot product

        # Compute dot product between all node pairs
        dot_products = torch.bmm(x, x.transpose(1, 2))  # (batch, vertex_count, vertex_count)
        approx_violated_constraints = dot_products * adj_matrix

        targets = torch.zeros_like(dot_products)

        loss = self.mse_loss(approx_violated_constraints, targets)

        return loss


class CustomCUTLossDotEXP(CustomGCOLLoss):
    def __init__(self, temp=1):
        super(CustomCUTLossDotEXP, self).__init__()
        self.temp = temp

    def constraint_approx(self, x, adj_matrix):
        """
        x has size (batch_size, seq_len (city_count), embed_size (city_count))
        x_ij is 1 if the city j is the ith city in the tour
        calculate the loss as an approximate of constraint violation (one single all diff constraint)
        """

        dot_products = torch.bmm(x, x.transpose(1, 2))  # (batch, vertex_count, vertex_count)
        approx_violated_constraints = dot_products * adj_matrix

        loss = torch.exp(self.temp * approx_violated_constraints)
        loss = torch.sum(loss)  # Sum over the batch dimension

        return loss


class CustomNRLossMSE(nn.Module):

    def __init__(self, hard=False):
        super(CustomNRLossMSE, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.lambda_alldiff = 200
        self.lambda_ineq = 1

        self.hard = hard

    def forward(self, predictions):

        batch_size, _, num_nurses = predictions.size()
        predictions = predictions.view(batch_size, -1, 3, 3, num_nurses)
        batch_size, num_days, shifts_per_day, nurses_per_shift, num_nurses = predictions.size()

        # constraint 1: day all diff
        predictions = predictions.view(batch_size * num_days, -1, num_nurses)
        # (batch_size*num_days, num_shifts*nurse_per_shift, num_nurses) -> (batch_size*num_days, num_nurses)
        day_sum_nurse = predictions.sum(dim=1)
        penalty_above_1 = F.relu(day_sum_nurse - 1)  # shape = (batch_size, embed_size)
        loss = penalty_above_1.pow(2).mean()
        binarization_term = (day_sum_nurse * (1.0 - day_sum_nurse)).mean()
        loss = loss + 0.1 * binarization_term

        # constraint 2: inequality
        predictions = predictions.view(batch_size, num_days, shifts_per_day, nurses_per_shift, num_nurses)
        last_shift = predictions[:, :-1, -1, :, :].sum(dim=-2) #(batch_size, num_days-1, num_nurses)
        first_shift = predictions[:, 1:, 0, :, :].sum(dim=-2) #(batch_size, num_days-1, num_nurses)
        shift_overlap = last_shift * first_shift
        shift_overlap = shift_overlap.sum(dim=-1)  #(batch_size, num_days-1, num_nurses) -> (batch_size, num_days-1)
        ineq_loss = self.mse_loss(shift_overlap, torch.zeros_like(shift_overlap))

        loss = self.lambda_alldiff * loss + self.lambda_ineq * ineq_loss

        return loss