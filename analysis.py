import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


def run_analysis_sudoku(trainer, device, test_iters=None, datasets=None):
    test_iters = [1, 2, 5, 10, 50, 100, 200, 500, 1000, 2000, 5000, 10000] if test_iters is None else test_iters
    datasets = ["47_64", "39_50"] if datasets is None else datasets

    results_list = []
    for dataset_name in datasets:
        with open(f'data/sudoku/sudoku_{dataset_name}_test.pt', 'rb') as f:
            x_in = torch.load(f, weights_only=True)
        with open(f'data/sudoku/labels_{dataset_name}_test.pt', 'rb') as f:
            y_in = torch.load(f, weights_only=True)
        x, y, constraint_graphs, is_input = trainer.csp_task.process_inputs(x_in.to(torch.float), y_in.to(torch.float))
        x, y, constraint_graphs, is_input = x.to(device), y.to(device), constraint_graphs.to(device), is_input.to(
            device)

        sudoku_ood = TensorDataset(x, y, constraint_graphs, is_input)
        oodloader = DataLoader(sudoku_ood, batch_size=1000, shuffle=False)

        instance_solved, run_time = trainer.evaluate_by_iters(
            oodloader, f'test_{dataset_name}', test_iters)
        for test_iter in test_iters:
            # Append results to the list
            results_list.append({
                "model_name": trainer.model_name,
                "dataset_name": f"sudoku_{dataset_name}",
                "test_iter": test_iter,
                "instance_accuracy": instance_solved[test_iter],
                "run_time": run_time[test_iter]
            })

        # Convert the results list into a DataFrame
        results_df = pd.DataFrame(results_list)
        results_df.to_csv(f"results/{trainer.model_name}")


def run_analysis_graph_coloring(trainer, colors, vertices_counts, candidate_pool=1):
    test_timeouts = [1, 2, 3, 4, 5, 10]
    results_list = []

    for vertices in vertices_counts:
        solved_sequential = {}
        time_sequential = {}
        iters_used = {}

        for test_iter in test_timeouts:
            solved_sequential[test_iter] = 0
            time_sequential[test_iter] = 0
            iters_used[test_iter] = 0

        inputs_path_test = f"data/graph_coloring/graphcoloring_{vertices}_vert_{colors}_color_test.pt"
        graph_adj_mat_tests = torch.load(inputs_path_test, weights_only=True)
        num_instances = graph_adj_mat_tests.size(0)

        for i in range(num_instances):  # for graph coloring, we solve one instance at a time for a fair comparison

            graph_adj_mat_test = graph_adj_mat_tests[i].unsqueeze(0)
            inputs_test = torch.zeros(graph_adj_mat_test.size(0), vertices, colors)
            var_inds_test = torch.ones(graph_adj_mat_test.size(0), vertices, dtype=torch.bool)
            batch_size, seq_len, _ = graph_adj_mat_test.size()
            constraint_graphs = graph_adj_mat_test.bool() + torch.eye(seq_len, seq_len, dtype=torch.bool).repeat(batch_size, 1, 1)

            if candidate_pool > 1:
                constraint_graphs = constraint_graphs.repeat(candidate_pool, 1, 1)
                inputs_test = inputs_test.repeat(candidate_pool, 1, 1)
                var_inds_test = var_inds_test.repeat(candidate_pool, 1, 1)

            test_dataset = TensorDataset(inputs_test, constraint_graphs, var_inds_test)
            oodloader = DataLoader(test_dataset, batch_size=candidate_pool, shuffle=True)

            instance_accuracy, run_time, iters_used_ins = trainer.evaluate_by_time_single_instance(oodloader, f'test_{vertices}_{colors}', test_timeouts)
            for test_iter in test_timeouts:
                solved_sequential[test_iter] += instance_accuracy[test_iter]
                time_sequential[test_iter] += run_time[test_iter]
                iters_used[test_iter] += iters_used_ins[test_iter]

        for test_iter in test_timeouts:
            # Append results to the list
            results_list.append({
                # "task_id": i,
                "model_name": trainer.model_name,
                "dataset_name": f"graph_coloring_{vertices}n_{colors}c",
                "test_timeout": test_iter,
                "iters_used": iters_used[test_iter] / num_instances,
                "correct_instances": solved_sequential[test_iter] / num_instances,
                "average_run_time": time_sequential[test_iter] / num_instances,
            })

        results_df = pd.DataFrame(results_list)

        results_df.to_csv(f"results/{trainer.model_name}_{candidate_pool}")


def run_analysis_nurse_scheduling(trainer, nurses):
    test_iters = [1, 2, 5, 10, 15, 30, 50, 100, 200]
    days = [20]
    spd, nps = 3, 3

    results_list = []

    for day in days:
        inputs_path_test = f"data/nurse_scheduling/nurseschedulesat_{day}days_{spd}spd_{nps}nps_{nurses}nurses_test.pt"

        with open(inputs_path_test, 'rb') as f:
            x_in = torch.load(f, weights_only=True).float()
        x_in = x_in.view(x_in.shape[0], -1, nurses)
        var_ind = x_in.sum(dim=-1, keepdim=True).int() == 0
        var_ind = var_ind.squeeze()
        constraint_graph = trainer.csp_task.get_constraint_graph_adj_mat(day, spd, nps)
        constraint_graph = constraint_graph.repeat(x_in.shape[0], 1, 1)
        dataset = TensorDataset(x_in, constraint_graph, var_ind)
        oodloader = DataLoader(dataset, batch_size=1000, shuffle=False)

        epoch_accuracy, time_elapsed = trainer.evaluate_by_iters(oodloader, f'test_{nurses}_{day}', test_iters)
        for test_iter in test_iters:
            # Append results to the list
            results_list.append({
                "model_name": trainer.model_name,
                "dataset_name": f"nr_{nurses}n_{day}d",
                "test_iter": test_iter,
                "run_time": time_elapsed[test_iter],
                "correct_instances": epoch_accuracy[test_iter],
            })

        results_df = pd.DataFrame(results_list)
        results_df.to_csv(f"results/{trainer.model_name}")

def run_analysis_CUT_GSET(trainer, candidate_pool=1):
    GSETS = ['G1', 'G2', 'G3', 'G4', 'G5', 'G14', 'G15', 'G16', 'G17', 'G22', 'G23', 'G24', 'G25', 'G26', 'G35', 'G36',
             'G37', 'G38', 'G43', 'G44', 'G45', 'G46', 'G47', 'G48', 'G49', 'G50', 'G51', 'G52', 'G53', 'G54', 'G55',
             'G58',
             'G60', 'G63', 'G70']

    GSETS_800 = ['G1', 'G2', 'G3', 'G4', 'G5', 'G14', 'G15', 'G16', 'G17']
    GSETS_1k = ['G43', 'G44', 'G45', 'G46', 'G47', 'G51', 'G52', 'G53', 'G54']
    GSETS_2k = ['G22', 'G23', 'G24', 'G25', 'G26', 'G35', 'G36', 'G37', 'G38']
    GSETS_3k = ['G48', 'G49', 'G50', 'G55', 'G58', 'G60', 'G63', 'G70']

    GSETSBEST = {'G1': 11624, 'G2': 11620, 'G3': 11622, 'G4': 11646, 'G5': 11631, 'G6': 2178, 'G7': 2006, 'G8': 2005,
                 'G9': 2054, 'G10': 2000, 'G11': 564, 'G12': 556, 'G13': 582, 'G14': 3064, 'G15': 3050, 'G16': 3052,
                 'G17': 3047,
                 'G18': 992, 'G19': 906, 'G20': 941, 'G21': 931, 'G22': 13359, 'G23': 13344, 'G24': 13337, 'G25': 13340,
                 'G26': 13328, 'G27': 3341, 'G28': 3298, 'G29': 3405, 'G30': 3413, 'G31': 3310, 'G32': 1410,
                 'G33': 1382,
                 'G34': 1384, 'G35': 7687, 'G36': 7680, 'G37': 7691, 'G38': 7688, 'G39': 2408, 'G40': 2400, 'G41': 2405,
                 'G42': 2481, 'G43': 6660, 'G44': 6650, 'G45': 6654, 'G46': 6649, 'G47': 6657, 'G48': 6000, 'G49': 6000,
                 'G50': 5880, 'G51': 3848, 'G52': 3851, 'G53': 3850, 'G54': 3852, 'G55': 10299, 'G56': 4017,
                 'G57': 3494,
                 'G58': 19293, 'G59': 6086, 'G60': 14188, 'G61': 5796, 'G62': 4870, 'G63': 27045, 'G64': 8751,
                 'G65': 5562,
                 'G66': 6364, 'G67': 6950, 'G70': 9591, 'G72': 7006, 'G77': None, 'G81': None,
                 "GSET_800": 7817.333333333333, "GSET_1K": 5407.888888888889, "GSET_2K": 10828.222222222223,
                 "GSET_3K": 12287.0}

    test_timeouts = [1, 2, 3, 4, 5, 10, 60, 180]

    results_list = []

    time_sequential = {}
    loss = {}
    iters_used = {}
    best_cuts = {}

    for test_iter in test_timeouts:
        time_sequential[test_iter] = 0
        loss[test_iter] = {}
        iters_used[test_iter] = {}
        best_cuts[test_iter] = {}
        for gsetname in GSETS:
            loss[test_iter][gsetname] = 0
            iters_used[test_iter][gsetname] = 0
            best_cuts[test_iter][gsetname] = 0

    for gsetname in GSETS:
        inputs_path_test = f"data/maxcut/GSET/{gsetname}.pt"
        graph_adj_mat_test = torch.load(inputs_path_test, weights_only=True)

        vertices = graph_adj_mat_test.size(1)
        inputs_test = torch.zeros(graph_adj_mat_test.size(0), vertices, 2)
        var_inds_test = torch.ones(graph_adj_mat_test.size(0), vertices, dtype=torch.bool)
        batch_size, seq_len, _ = graph_adj_mat_test.size()
        constraint_graphs = graph_adj_mat_test.bool() + torch.eye(seq_len, seq_len, dtype=torch.bool).repeat(batch_size,
                                                                                                             1, 1)
        if candidate_pool > 1:
            constraint_graphs = constraint_graphs.repeat(candidate_pool, 1, 1)
            inputs_test = inputs_test.repeat(candidate_pool, 1, 1)
            var_inds_test = var_inds_test.repeat(candidate_pool, 1, 1)

        test_dataset = TensorDataset(inputs_test, constraint_graphs, var_inds_test)
        oodloader = DataLoader(test_dataset, batch_size=candidate_pool, shuffle=True)


        epoch_cut_ins, time_elapsed_ins, iters_used_ins = trainer.evaluate_by_time_single_instance(oodloader, f'test_{vertices}', test_timeouts)
        for test_iter in test_timeouts:
            time_sequential[test_iter] += time_elapsed_ins[test_iter]
            iters_used[test_iter][gsetname] += iters_used_ins[test_iter]
            best_cuts[test_iter][gsetname] = epoch_cut_ins[test_iter]

        print(f"Instance {gsetname} best cut in 180 seconds is {best_cuts[180][gsetname]}")

    for test_iter in test_timeouts[:-1]:

        loss_800 = sum([loss[test_iter][gse] for gse in GSETS_800]) / len(GSETS_800)
        loss_1k = sum([loss[test_iter][gse] for gse in GSETS_1k]) / len(GSETS_1k)
        loss_2k = sum([loss[test_iter][gse] for gse in GSETS_2k]) / len(GSETS_2k)
        loss_3k = sum([loss[test_iter][gse] for gse in GSETS_3k]) / len(GSETS_3k)

        iters_800 = sum([iters_used[test_iter][gse] for gse in GSETS_800]) / len(GSETS_800)
        iters_1k = sum([iters_used[test_iter][gse] for gse in GSETS_1k]) / len(GSETS_1k)
        iters_2k = sum([iters_used[test_iter][gse] for gse in GSETS_2k]) / len(GSETS_2k)
        iters_3k = sum([iters_used[test_iter][gse] for gse in GSETS_3k]) / len(GSETS_3k)

        cuts_800 = sum([best_cuts[test_iter][gse] - GSETSBEST[gse] for gse in GSETS_800]) / len(GSETS_800)
        cuts_1k = sum([best_cuts[test_iter][gse] - GSETSBEST[gse] for gse in GSETS_1k]) / len(GSETS_1k)
        cuts_2k = sum([best_cuts[test_iter][gse] - GSETSBEST[gse] for gse in GSETS_2k]) / len(GSETS_2k)
        cuts_3k = sum([best_cuts[test_iter][gse] - GSETSBEST[gse] for gse in GSETS_3k]) / len(GSETS_3k)


        results_list.append({
            # "task_id": i,
            "model_name": f"{trainer.model_name}-pool{candidate_pool}",
            **{f"best_cuts_{gs}": best_cuts[test_iter][gs] for gs in GSETS},
            "dataset_name": f"GSET",
            "test_timeout": test_iter,
            "loss_800": loss_800,
            "loss_1k": loss_1k,
            "loss_2k": loss_2k,
            "loss_3k": loss_3k,
            "iters_800": iters_800,
            "iters_1k": iters_1k,
            "iters_2k": iters_2k,
            "iters_3k": iters_3k,
            "best_cuts_800": cuts_800,
            "best_cuts_1k": cuts_1k,
            "best_cuts_2k": cuts_2k,
            "best_cuts_3k": cuts_3k,
            "average_run_time": time_sequential[test_iter] / len(GSETS),
        })

    results_df = pd.DataFrame(results_list)
    results_df.to_csv(f"maxcut/results/{trainer.model_name}_pool{candidate_pool}")

