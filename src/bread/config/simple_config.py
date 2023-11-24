
import os
from importlib import import_module

frame_graph = {
    "num_epoches": 1,
    "edge_dim": 5,
    "input_dimention": 6,
    "output_dimention": 8,
    "hidden_channels": [16,64],
    "num_heads": 4,
    "add_self_loops": True,
    "dropout": 0.005,
    "lr": 0.01,
    "lr_decay_factor": 0.5,
    "loss_function": "mse",
}

########################################################################################################################
# Configuration that will be loaded
########################################################################################################################

configuration = {
    "frame_graph": {
        **frame_graph,
    },
    "use_wandb": False,
    "memory": 3000,
    "stop": {"training_iteration": 1000, 'patience': 10},
    "scheduler": None,
    "search_alg": None,
}