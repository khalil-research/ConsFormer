# ConsFormer

Official implementation of the ICML 2025 paper 

[*Self-Supervised Transformers as Iterative Solution Improvers for Constraint Satisfaction*](https://arxiv.org/abs/2502.15794).

![ConsFormer Architecture](consformer_architecture.png)


## Code Structure

```
.
├── requirements.txt
├── analysis.py
├── main_graph_coloring.py
├── main_maxcut.py
├── main_nurse_scheduling.py
├── main_sudoku.py
├── consformer/
│   ├── criterion.py
│   ├── csptask.py
│   ├── embeddings.py
│   ├── modules.py
│   ├── solvers.py
│   └── trainer.py
├── data/
├── results/
└── saved_models/

```

**`analysis.py`**  
Model analysis on OOD instances.

**`main_{task}.py`**  
Entry-point scripts for each task (`graph_coloring`, `maxcut`, `nurse_scheduling`, `sudoku`).

**`consformer/criterion.py`**  
Defines loss functions and accuracy/constraint‐violation metrics.

**`consformer/csptask.py`**  
Task-specific data loading, preprocessing and evaluation.

**`consformer/embeddings.py`**  
Embedding layers.

**`consformer/modules.py`**  
Core Transformer building blocks.

**`consformer/solvers.py`**  
The full “ConsFormer” solver class.

**`consformer/trainer.py`**  
Manages training and testing loops.

**`data/`**  
Contains datasets.

**`results/`**  
Output directory for evaluation summaries.

**`saved_models/`**  
Stores checkpointed model weights.

## Usage
Each problem can be run by executing its respective `main` file.

For example, to run the model on Sudoku:
```
python main_sudoku.py
```
You can modify the model/training set up with command line args, for example:
```
python main_sudoku.py --threshold 0.5 --dropout 0.1 --head-count 3 --layer-count 4 --hidden-size 128 --ape-dim 0 --optimizer AdamW --epochs 1000 --loss DecomposedMSE
```
read about the args for each task by using `--help`.

## Data

You can download the data we used [here](https://drive.google.com/file/d/1WP-g_7yXl0yKSbQkUAaICYVHEpJF7dgH/view?usp=sharing).

## Citation

If you use this code in your research, please cite:
```
@inproceedings{xu:icml25,
  title = {Self-Supervised Transformers as Iterative Solution Improvers for Constraint Satisfaction},
  author = {Yudong Xu and Wenhao Li and Scott Sanner and Elias Boutros Khalil},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning (ICML-25)},
  address = {Vancouver, Canada},
  year = {2025},
  url_arxiv = {https://arxiv.org/abs/2502.15794}
}
```

## Contact

Please reach out to [Yudong Will Xu](https://xuwil.github.io/) if you have any questions. 
Slides and Poster of this paper can also be found at this link.

