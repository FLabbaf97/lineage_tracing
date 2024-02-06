
from bread.data import *
from bread.algo.tracking import AssignmentDataset, GNNTracker, AssignmentClassifier, GraphLoader, accuracy_assignment, f1_assignment

from glob import glob
from pathlib import Path
import datetime, json, argparse
import torch, os
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import KFold
from skorch.dataset import ValidSplit
from skorch.callbacks import LRScheduler, Checkpoint, ProgressBar, Checkpoint, EpochScoring, EarlyStopping, WandbLogger
from sklearn.model_selection import ParameterGrid

from trainutils import SaveHyperParams, seed
from importlib import import_module

import wandb

if __name__ == '__main__':
        
    WANDB_API_KEY="e0f887ce4be7bebfe48930ffcff4027f49b02425"
    os.environ['WANDB_API_KEY'] = WANDB_API_KEY
    os.environ['WANDB_CONSOLE'] = "off"
    os.environ['WANDB_JOB_TYPE'] = 'features_test'

    config_file = 'config_10f'

    config = import_module("bread.config." + config_file).configuration
    pretty_config = json.dumps(config, indent=4)
    print(pretty_config)
    train_config = config.get('train_config')

    data_dir = Path(train_config["ass_graphs"])
    dataset_filepaths = {
        'colony_1234__dt_12345678__t_all': list(sorted(glob(str(data_dir / 'colony00[1-4]_segmentation__assgraph__dt_00[1-8]__*.pt')))),
        'colony_5__dt_12345678__t_all': list(sorted(glob(str(data_dir / 'colony005_segmentation__assgraph__dt_00[1-8]__*.pt')))),
        'colony_1234__dt_1234__t_all': list(sorted(glob(str(data_dir / 'colony00[1-4]_segmentation__assgraph__dt_00[1-4]__*.pt')))),
        'colony_5__dt_1234__t_all': list(sorted(glob(str(data_dir / 'colony005_segmentation__assgraph__dt_00[1-4]__*.pt')))),
        'colony_12345__dt_1234__t_all': list(sorted(glob(str(data_dir / 'colony00[1-5]_segmentation__assgraph__dt_00[1-4]__*.pt')))),
        'colony_012345__dt_1234__t_all': list(sorted(glob(str(data_dir / 'colony00[0-5]_segmentation__assgraph__dt_00[1-4]__*.pt')))),
        'colony_0123__dt_1234__t_all': list(sorted(glob(str(data_dir / 'colony00[0-3]_segmentation__assgraph__dt_00[1-4]__*.pt')))),
    }
    dataset_regex = {
        'colony_1234__dt_12345678__t_all': "^colony00(1|2|3|4)_segmentation__assgraph__dt_00(1|2|3|4|5|6|7|8)__.*pt$",
        'colony_5__dt_12345678__t_all': "^colony005_segmentation__assgraph__dt_00(1|2|3|4|5|6|7|8)__.*pt$",
        'colony_1234__dt_1234__t_all': "^colony00(1|2|3|4)_segmentation__assgraph__dt_00(1|2|3|4)__.*pt$",
        'colony_5__dt_1234__t_all': "^colony005_segmentation__assgraph__dt_00(1|2|3|4)__.*pt$",
        'colony_12345__dt_1234__t_all': "^colony00(1|2|3|4|5)_segmentation__assgraph__dt_00(1|2|3|4)__.*pt$",
        'colony_012345__dt_1234__t_all': "^colony00(0|1|2|3|4|5)_segmentation__assgraph__dt_00(1|2|3|4)__.*pt$",
        'colony_01234__dt_1234__t_all': "^colony00(0|1|2|3|4)_segmentation__assgraph__dt_00(1|2|3|4)__.*pt$",
        'colony_0123__dt_1234__t_all': "^colony00(0|1|2|3)_segmentation__assgraph__dt_00(1|2|3|4)__.*pt$",
        'colony_6__dt_1234__t_all': "^colony006_segmentation__assgraph__dt_00(1|2|3|4)__.*åpt$",
        'colony_6__dt_1__t_all': "^colony006_segmentation__assgraph__dt_001__.*pt$",
        'colony_0123__dt_1234__t_all': "^colony00(0|1|2|3)_segmentation__assgraph__dt_00(1|2|3|4)__.*pt$",
        'colony_0__dt_1__t_all': "^colony000_segmentation__assgraph__dt_001__.*pt$",
        'colony_6__dt_1234__t_all': "^colony006_segmentation__assgraph__dt_00(1|2|3|4)__.*pt$",
        'colony_56__dt_1234__t_all': "^colony00(5|6)_segmentation__assgraph__dt_00(1|2|3|4)__.*pt$",
    }
    dataset_name_train = 'colony_01234__dt_1234__t_all'



    resultdir = Path(f'{train_config["result_dir"]}/{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print("-- result directory --")
    print(resultdir)
    os.makedirs(resultdir, exist_ok=True)
    with open(resultdir / 'metadata.json', 'w') as file:
        json.dump(config, file)

    # filter out files that are too large
    file_array = []
    import re
    regex = dataset_regex[train_config["dataset"]]
    for filename in os.listdir(train_config["ass_graphs"]):
        if re.search(regex, filename):
            file_array.append(os.path.join(train_config["ass_graphs"], filename))
    file_array = [ filepath for filepath in file_array if os.stat(filepath).st_size/2**20 < train_config["filter_file_mb"] ]
    dataset = AssignmentDataset(file_array)

    print('-- training dataset --')
    print(dataset)

    seed(42)
    dataset = dataset.shuffle()

    cv = None  
    if train_config["cv"] is not None:
        cv = ValidSplit(
            # This generates one train and validation dataset for the entire fit
            cv=train_config["cv"],
            stratified=False,  # Since we are using y=None
        )


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    # Create a wandb Run
    # wandb_run = wandb.init()
    # # Log hyper-parameters 
    # wandb_run.config.update(config)
    # net = AssignmentClassifier(
    # 	GNNTracker,
    # 	module__num_node_attr=len(dataset.node_attr),
    # 	module__num_edge_attr=len(dataset.edge_attr),
    # 	module__dropout_rate=train_config["dropout_rate"],
    # 	module__encoder_hidden_channels=train_config["encoder_hidden_channels"],
    # 	module__encoder_num_layers=train_config["encoder_num_layers"],
    # 	module__conv_hidden_channels=train_config["conv_hidden_channels"],
    # 	module__conv_num_layers=train_config["conv_num_layers"],
    # 	module__num_classes=1,  # fixed, we do binary classification
    # 	max_epochs=train_config["max_epochs"],
    # 	device=device,

    # 	criterion=torch.nn.BCEWithLogitsLoss(
    # 		# attribute more weight to the y == 1 samples, because they are more rare
    # 		pos_weight=torch.tensor((dataset.num_class_positive + dataset.num_class_negative)/dataset.num_class_positive)
    # 	),

    # 	optimizer=torch.optim.Adam,
    # 	optimizer__lr=train_config["lr"],
    # 	optimizer__weight_decay=train_config["weight_decay"],  # L2 regularization

    # 	iterator_train=GraphLoader,
    # 	iterator_valid=GraphLoader,

    # 	iterator_train__shuffle=True,
    # 	iterator_valid__shuffle=False,
    # 	batch_size=1,

    # 	train_split=cv,

    # 	callbacks=[
    # 		LRScheduler(policy='StepLR', step_every='batch', step_size=train_config["step_size"], gamma=train_config["gamma"]),
    # 		Checkpoint(monitor='valid_loss_best', dirname=resultdir, f_pickle='pickle.pkl'),
    # 		SaveHyperParams(dirname=resultdir),
    # 		EarlyStopping(patience=train_config["patience"]),
    # 		ProgressBar(detect_notebook=False),
    # 		WandbLogger(wandb_run, save_model=False),
    # 		EpochScoring(scoring=f1_assignment, lower_is_better=False, name='valid_f1_ass'),
    # 	],
    # )

    # experiment (A)
    # grid = ParameterGrid({
    # 	'module__encoder_hidden_channels': [100, 60, 20],
    # 	'module__encoder_num_layers': [8, 5, 3],
    # 	'module__conv_hidden_channels': [120, 100, 60],
    # 	'module__conv_num_layers': [10, 8, 4],
    # })
    # experiment (B)
    # grid = ParameterGrid({
    # 	'module__dropout_rate': [1e-2, 1e-4, 1e-6],
    # 	'optimizer__weight_decay': [1e-2, 1e-4, 1e-6, 0.0],
    # })
    # experiment (C)
    # grid = ParameterGrid({
    # 	'module__dropout_rate': [1e-5, 1e-7, 0.0],
    # 	'optimizer__weight_decay': [1e-3, 1e-5, 0.0],
    # 	'module__encoder_hidden_channels': [140, 110, 80],
    # })
    # experiment (D)

    grid = ParameterGrid({
        'encoder_hidden_channels': [120, 60, 20],
        'encoder_num_layers': [4, 2],
        'conv_hidden_channels': [120, 100, 60],
        'conv_num_layers': [1, 2, 5, 8],
        'dropout_rate': [1e-7, 1e-3],
    })
    results = []

    resultroot = resultdir

    for params in tqdm(grid):
        params_str = ' '.join([f'{k}={v}' for k, v in params.items()])
        tqdm.write(params_str)
        updated_dict = {key: params[key] if key in params else value for key, value in train_config.items()}
        train_config = updated_dict

        kfold = KFold(n_splits=3, shuffle=True, random_state=42)
        for kth, (idx_train, idx_val) in enumerate(kfold.split(dataset)):
            tqdm.write(f'split {kth+1}/{kfold.n_splits}')

            resultdir = resultroot / params_str / f'split {kth+1}'
            seed(42)

            # net.set_params(
            # 	train_split=lambda dataset: (dataset[idx_train], dataset[idx_val]),
            # 	verbose=1,
            # 	callbacks__WandbLogger=WandbLogger(wandb_run, save_model=False),
            # 	callbacks__Checkpoint__dirname=resultdir,
            # 	callbacks__SaveHyperParams__dirname=resultdir,
            # 	**params,
            # )

            # Create a wandb Run
            wandb_run = wandb.init()
            # Log hyper-parameters 
            wandb_run.config.update(train_config)


            net = AssignmentClassifier(
                GNNTracker,
                module__num_node_attr=len(dataset.node_attr),
                module__num_edge_attr=len(dataset.edge_attr),
                module__dropout_rate=train_config["dropout_rate"],
                module__encoder_hidden_channels=train_config["encoder_hidden_channels"],
                module__encoder_num_layers=train_config["encoder_num_layers"],
                module__conv_hidden_channels=train_config["conv_hidden_channels"],
                module__conv_num_layers=train_config["conv_num_layers"],
                module__num_classes=1,  # fixed, we do binary classification
                max_epochs=train_config["max_epochs"],
                device=device,

                criterion=torch.nn.BCEWithLogitsLoss(
                    # attribute more weight to the y == 1 samples, because they are more rare
                    pos_weight=torch.tensor((dataset.num_class_positive + dataset.num_class_negative)/dataset.num_class_positive)
                ),

                optimizer=torch.optim.Adam,
                optimizer__lr=train_config["lr"],
                optimizer__weight_decay=train_config["weight_decay"],  # L2 regularization

                iterator_train=GraphLoader,
                iterator_valid=GraphLoader,

                iterator_train__shuffle=True,
                iterator_valid__shuffle=False,
                batch_size=1,

                train_split=cv,

                callbacks=[
                    LRScheduler(policy='StepLR', step_every='batch', step_size=train_config["step_size"], gamma=train_config["gamma"]),
                    Checkpoint(monitor='valid_loss_best', dirname=resultdir, f_pickle='pickle.pkl'),
                    SaveHyperParams(dirname=resultdir),
                    EarlyStopping(patience=train_config["patience"]),
                    ProgressBar(detect_notebook=False),
                    WandbLogger(wandb_run, save_model=False),
                    EpochScoring(scoring=f1_assignment, lower_is_better=False, name='valid_f1_ass'),
                ],
            )


            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
            torch.cuda.empty_cache()
            net.fit(dataset, y=None)

            d = { k.replace('module__', ''): v for k, v in params.items() }
            d.update({
                'valid_loss': min(net.history[:, 'valid_loss']),
                'valid_err': 1-min(net.history[:, 'valid_acc'])
            })
            results.append(d)


    df = pd.DataFrame(results)
    df.to_csv('results_C.csv')


    df\
        .groupby(list(df.drop(columns='valid_loss').columns))\
        [['valid_loss']]\
        .agg(['mean', 'std'])\
        .reset_index()\
        .nsmallest(20, ('valid_loss', 'mean'))
