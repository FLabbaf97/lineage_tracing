import os
from importlib import import_module
from pathlib import Path

# cell_graph_config = {
#     "cell_f_list": [
#         'area',
#         'r_equiv',
#         'r_maj',
#         'r_min',
#         'angel',
#         'ecc',
#     ],
#     "edge_f_list": [
#         "cmtocm_x",
#         "cmtocm_y",
#         "cmtocm_len",
#         "cmtocm_angle",
#         "contour_dist",
#     ],
#     "nn_threshold": 12,
#     "scale_length": 1.0,
#     "scale_time": 1.0,
#     "output_folder": "/mlodata1/hokarami/fari/bread/data/generated/test_tracking/cell_graphs_original",
#     "input_segmentations":[
#         # '/mlodata1/hokarami/fari/bread/data/segmentations/colony000_segmentation.h5',
#         '/mlodata1/hokarami/fari/bread/data/segmentations/colony001_segmentation.h5',
#         '/mlodata1/hokarami/fari/bread/data/segmentations/colony002_segmentation.h5',
#         '/mlodata1/hokarami/fari/bread/data/segmentations/colony003_segmentation.h5',
#         '/mlodata1/hokarami/fari/bread/data/segmentations/colony004_segmentation.h5',
#         '/mlodata1/hokarami/fari/bread/data/segmentations/colony005_segmentation.h5',
#         '/mlodata1/hokarami/fari/bread/data/segmentations/colony006_segmentation.h5',

#     ]
# }

# ass_graph_config = {
#     "cellgraph_dirs": [
#         # "/mlodata1/hokarami/fari/bread/data/generated/test_tracking/cell_graphs_original/colony000_segmentation/",
#         "/mlodata1/hokarami/fari/bread/data/generated/test_tracking/cell_graphs_original/colony001_segmentation/",
#         "/mlodata1/hokarami/fari/bread/data/generated/test_tracking/cell_graphs_original/colony002_segmentation/",
#         "/mlodata1/hokarami/fari/bread/data/generated/test_tracking/cell_graphs_original/colony003_segmentation/",
#         "/mlodata1/hokarami/fari/bread/data/generated/test_tracking/cell_graphs_original/colony004_segmentation/",
#         "/mlodata1/hokarami/fari/bread/data/generated/test_tracking/cell_graphs_original/colony005_segmentation/",
#         "/mlodata1/hokarami/fari/bread/data/generated/test_tracking/cell_graphs_original/colony006_segmentation/",

#     ],
#     "output_folder": "/mlodata1/hokarami/fari/bread/data/generated/test_tracking/ass_graphs_original",
#     "framediff_min": 1,
#     "framediff_max": 4,
#     "t1_max": -1,# default on -1
#     "t1_min": 0,
# }

train_config = {
    "ass_graphs": "/mlodata1/hokarami/fari/bread/data/generated/test_tracking/ass_graphs_original",
    "result_dir": "/mlodata1/hokarami/fari/bread/data/generated/test_tracking/results_original",
    "dataset": "colony_01234__dt_1234__t_all",
    "valid_dataset": '3',
    "min_file_kb": 1000,
    "filter_file_mb": 10, # filter training file sizes exceeding `filter_file_mb` MiB
    "dropout_rate": 0.0,
    "encoder_hidden_channels": 32,
    "encoder_num_layers": 2,
    "conv_hidden_channels": 32,
    "conv_num_layers": 2, 
    "max_epochs" : 40,
    "lr": 1e-3, 
    "weight_decay": 0.01, # (L2 regularization)
    "step_size": 10, # steplr step size in number of batches
    "gamma": 0.5, # steplr gamma
    "cv": 1,
    "patience": 4,
    "scoring": 'valid_f1_ass'
}

test_config = {
    "ass_graphs": "/mlodata1/hokarami/fari/bread/data/generated/test_tracking/ass_graphs_original",
    "result_dir": "/mlodata1/hokarami/fari/bread/data/generated/test_tracking/results_original",
    "dataset": "colony_6__dt_1234__t_all",
    "filter_file_mb": 20, # filter file sizes exceeding `filter_file_mb` MiB
}

########################################################################################################################
# Configuration that will be loaded
########################################################################################################################

configuration = {
    "train_config": {
        **train_config,
    },
    "test_config":{
        **test_config,
    },
    "use_wandb": False,
}