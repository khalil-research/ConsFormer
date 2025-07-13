import argparse
from torch import optim

from consformer.csptask import SudokuTask
from consformer.solvers import ConsFormer
from consformer.criterion import *
from consformer.trainer import Trainer
from analysis import *

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="parsing input arguments", prog='PROG', usage='%(prog)s [options]')

    parser.add_argument("--batch-size", type=int, default=512, required=False)
    parser.add_argument("--epochs", type=int, default=3000, required=False)
    parser.add_argument("--dropout", type=float, default=0.1, required=False)
    parser.add_argument("--data-range", type=str, choices=["39_50", "47_64"],
                        #  39_50 corresponds to the SATNet dataset, 47_64 corresponds to the RRN dataset
                        help="training set range of missing values from sudoku board", default="39_50", required=False)
    parser.add_argument("--loss", type=str, choices=["ABSE", "MSE", "DecomposedMSE"],
                        help="Penalty-based self-supervised loss for sudoku", default="MSE", required=False)
    parser.add_argument("--optimizer", type=str, choices=["RMSprop", "Adam", "AdamW"],
                        default="AdamW", required=False)
    parser.add_argument("--learning-rate", type=float, default=0.0001, required=False)
    parser.add_argument("--threshold", help="subset selection threshold",
                        #  0.7 means ~30% of variables will be included in the subset
                        type=float, default=0.7, required=False)
    parser.add_argument("--head-count", type=int, default=3, required=False)
    parser.add_argument("--layer-count", type=int, default=7, required=False)
    parser.add_argument("--hidden-size", type=int, default=128, required=False)
    parser.add_argument("--mixing-strategy", type=str, choices=["add", "no"], default="add", required=False)
    parser.add_argument("--ape-dim", type=int, default=2, choices=[0, 1, 2], required=False)
    parser.add_argument("--no-train", action="store_true",
                        help="if present, skip training and do analysis, assuming the model is already trained",
                        required=False)
    parser.add_argument("--rpe", type=str, choices=["learned", "mask", "no"], default="no",
                        help="if present use binary constrain graph as rpe for the attention", required=False)
    parser.add_argument("--no-gumbel", action="store_true",
                        help="if present, use softmax instead of gumbel-softmax for generating the solution")
    parser.add_argument("--tau", type=float, default=0.1, help="(gumbel)softmax temperature")

    return parser.parse_args()


def main():

    args = parse_args()

    # model params for sudoku
    input_size = 9
    output_size = 9
    embedding_size = args.hidden_size
    tf_hidden_size = args.hidden_size
    expand_size = tf_hidden_size
    head_count = args.head_count
    dropout = args.dropout
    num_layers = args.layer_count
    vocab_size = 9
    subset_threshold = args.threshold
    mixing_strategy = args.mixing_strategy
    ape_dim = args.ape_dim
    rpe = args.rpe
    no_gumbel = args.no_gumbel
    tau = args.tau

    # training params
    num_epochs = args.epochs
    learning_rate = args.learning_rate
    loss_function_name = args.loss
    optimizer_name = args.optimizer
    batch_size = args.batch_size

    loss_functions = {"ABSE": CustomSudokuLossABSE,
                      "MSE": CustomSudokuLossMSE,
                      "DecomposedMSE": CustomSudokuLossDecomposedMSE,
                      }

    optimizer_funcs = {"AdamW": optim.AdamW,
                       "RMSprop": optim.RMSprop,
                       "Adam": optim.Adam,
                       }

    model_name = f"sudoku-{rpe}rpe-{subset_threshold}-{args.data_range}-{ape_dim}dape-{mixing_strategy}mixer-{loss_function_name}-{'softmax' if no_gumbel else 'gumbel'}{tau}-{optimizer_name}-dropout{dropout}-embed{embedding_size}-{head_count}h{num_layers}l-bs{batch_size}-lr{learning_rate}"

    inputs_path_train = f"./data/sudoku/sudoku_{args.data_range}_train.pt"
    inputs_path_test = f"./data/sudoku/sudoku_{args.data_range}_test.pt"
    labels_path_train = f"./data/sudoku/labels_{args.data_range}_train.pt"
    labels_path_test = f"./data/sudoku/labels_{args.data_range}_test.pt"

    acc_func = CustomSudokuAccuracy()
    loss_func = loss_functions[loss_function_name]()
    sudoku_task = SudokuTask(loss_func, acc_func)

    train_loader, test_loader = sudoku_task.get_data_loaders(inputs_path_train, labels_path_train,
                                                             inputs_path_test, labels_path_test, batch_size)

    model = ConsFormer(input_size=input_size,
                       embedding_size=embedding_size,
                       hidden_size=tf_hidden_size,
                       output_size=output_size,
                       num_heads=head_count,
                       expand_size=expand_size,
                       drop_out=dropout,
                       num_layers=num_layers,
                       vocab_size=vocab_size,
                       subset_threshold=subset_threshold,
                       ape_dim=ape_dim,
                       mixing_strategy=mixing_strategy,
                       rpe=rpe,
                       tau=tau,
                       no_gumbel=no_gumbel,
                       )

    optimizer = optimizer_funcs[optimizer_name](model.parameters(), lr=learning_rate)

    # scheduler = ExponentialLR(optimizer, gamma=0.90)  # Exponential decay of learning rate
    # scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.00001)
    scheduler = None

    trainer = Trainer(
        model=model,
        csp_task=sudoku_task,
        device=device,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=num_epochs,
        optimizer=optimizer,
        scheduler=scheduler,
        learning_rate=learning_rate,
        log_interval=10,
        model_name=model_name,
    )

    if not args.no_train:
        trainer.train()
    trainer.model.load_state_dict(
        torch.load(f"saved_models/{model_name}_best", map_location=device, weights_only=True))
    run_analysis_sudoku(trainer, device, test_iters=[1,2,5,10])


if __name__ == '__main__':
    main()