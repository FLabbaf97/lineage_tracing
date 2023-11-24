import os
from importlib import import_module
from pathlib import Path

cell_graph_config = {
    "cell_f_list": [
        'fourier',
        10,
        True,
        [
            'area',
            'r_equiv',
            'r_maj',
            'r_min',
            'angel',
            'ecc',
            'maj_x',
            'maj_y',
            'min_x',
            'min_y',
        ]
    ],
    "edge_f_list": [
        "cmtocm_x",
        "cmtocm_y",
        "cmtocm_len",
        "cmtocm_angle",
        "contour_dist",
    ],
    "nn_threshold": 12,
    "scale_length": 1.0,
    "scale_time": 1.0,
    "output_folder": "/mlodata1/hokarami/fari/bread/data/generated/test_tracking/cell_graphs_fourier_10_and_f10_locality_True",
    "input_segmentations":[
        # '/mlodata1/hokarami/fari/bread/data/segmentations/colony000_segmentation.h5',
        '/mlodata1/hokarami/fari/bread/data/segmentations/colony001_segmentation.h5',
        '/mlodata1/hokarami/fari/bread/data/segmentations/colony002_segmentation.h5',
        '/mlodata1/hokarami/fari/bread/data/segmentations/colony003_segmentation.h5',
        '/mlodata1/hokarami/fari/bread/data/segmentations/colony004_segmentation.h5',
        '/mlodata1/hokarami/fari/bread/data/segmentations/colony005_segmentation.h5',
        '/mlodata1/hokarami/fari/bread/data/segmentations/colony006_segmentation.h5',
    ]
}

ass_graph_config = {
    "cellgraph_dirs": [
        "/mlodata1/hokarami/fari/bread/data/generated/test_tracking/cell_graphs_fourier_10_and_f10_locality_True/colony000_segmentation/",
        "/mlodata1/hokarami/fari/bread/data/generated/test_tracking/cell_graphs_fourier_10_and_f10_locality_True/colony001_segmentation/",
        "/mlodata1/hokarami/fari/bread/data/generated/test_tracking/cell_graphs_fourier_10_and_f10_locality_True/colony002_segmentation/",
        "/mlodata1/hokarami/fari/bread/data/generated/test_tracking/cell_graphs_fourier_10_and_f10_locality_True/colony003_segmentation/",
        "/mlodata1/hokarami/fari/bread/data/generated/test_tracking/cell_graphs_fourier_10_and_f10_locality_True/colony004_segmentation/",
        "/mlodata1/hokarami/fari/bread/data/generated/test_tracking/cell_graphs_fourier_10_and_f10_locality_True/colony005_segmentation/",
        "/mlodata1/hokarami/fari/bread/data/generated/test_tracking/cell_graphs_fourier_10_and_f10_locality_True/colony006_segmentation/",

    ],
    "output_folder": "/mlodata1/hokarami/fari/bread/data/generated/test_tracking/ass_graphs_fourier_10_and_f10_locality_True",
    "framediff_min": 1,
    "framediff_max": 4,
    "t1_max": -1, # default on -1
    "t1_min": 0
}

train_config = {
    "ass_graphs": "/mlodata1/hokarami/fari/bread/data/generated/test_tracking/ass_graphs_fourier_10_and_f10_locality_True",
    "result_dir": "/mlodata1/hokarami/fari/bread/data/generated/test_tracking/results_fourier_10_and_f10_locality_True",
    "dataset": "colony_01234__dt_1234__t_all",
    "valid_dataset": '3',
    "min_file_kb": 100,
    "filter_file_mb": 50, # filter training file sizes exceeding `filter_file_mb` MiB
    "dropout_rate": 1e-7,
    "encoder_hidden_channels": 120,
    "encoder_num_layers": 4,
    "conv_hidden_channels": 120,
    "conv_num_layers": 5, 
    "max_epochs" : 50,
    "lr": 1e-3, 
    "weight_decay": 0.01, # (L2 regularization)
    "step_size": 512, # steplr step size in number of batches
    "gamma": 0.5, # steplr gamma
    "cv": 3,
    "patience": 6,
}

test_config = {
    "ass_graphs": "/mlodata1/hokarami/fari/bread/data/generated/test_tracking/ass_graphs_fourier_10_and_f10_locality_True",
    "result_dir": "/mlodata1/hokarami/fari/bread/data/generated/test_tracking/results_fourier_10_and_f10_locality_True",
    "dataset": "colony_56__dt_1234__t_all",
    "filter_file_mb": 20, # filter file sizes exceeding `filter_file_mb` MiB
}
########################################################################################################################
# Configuration that will be loaded
########################################################################################################################

configuration = {
    "cell_graph_config": {
        **cell_graph_config,
    },
    "ass_graph_config": {
        **ass_graph_config,
    },
    "train_config": {
        **train_config,
    },
    "test_config": {
        **test_config,
    },
    "use_wandb": True,
}