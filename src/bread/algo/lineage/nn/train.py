if __name__ == '__main__':
	import argparse
	from pathlib import Path
	import logging
	logging.basicConfig()
	logger = logging.getLogger('trainer')
	logger.setLevel(logging.INFO)

	parser = argparse.ArgumentParser('bread.algo.lineage.nn.train', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--model', dest='model_str', required=True, choices=['deep', 'logistic'], help='model to train')
	parser.add_argument('--num-hidden', dest='num_hidden', type=int, default=2, help='number of hidden layers')
	parser.add_argument('--size-hidden', dest='size_hidden', type=int, default=128, help='size of the hidden layers')
	parser.add_argument('--dropout', dest='dropout', type=float, default=0.1, help='dropout probability')
	parser.add_argument('--lr', dest='lr', type=float, default=1e-4, help='learning rate')
	parser.add_argument('--lr-decay', dest='lr_decay', type=float, default=0.97, help='learning rate decay')  # decay in total 10**(-1) in total after 100 epochs, 10**(-1/100) = 0.977, and 0.977**100 = 0.0977
	parser.add_argument('--seed', dest='seed', type=int, default=0, help='random seed')
	parser.add_argument('--batch-size', dest='batch_size', type=int, default=64, help='batch size')
	parser.add_argument('--num-epochs', dest='num_epochs', type=int, default=100, help='number of epochs')
	parser.add_argument('--data-train', dest='fp_data_train', type=Path, default=Path('data/generated/data_train.pt'), help='path to the training/validation data (as exported by bread.algo.lineage.nn.preprocess)')
	parser.add_argument('--data-test', dest='fp_data_test', type=Path, default=Path('data/generated/data_test.pt'), help='path to the testing data (as exported by bread.algo.lineage.nn.preprocess)')
	parser.add_argument('--num-folds', dest='num_folds', type=int, default=4, help='number of folds for k-fold validation. set to 0 to disable')
	# parser.add_argument('--equalize', dest='equalize', action='store_true', help='equalize frequency of classes in the dataset during training and validating')
	parser.add_argument('--dir-logs', dest='dir_logs', type=Path, default=Path('data/generated/pytorch'), help='directory to use for logs')

	args = parser.parse_args()

	logger.info(args)

	# import sys
	# sys.exit(0)

	import pytorch_lightning as pl
	from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
	from pytorch_lightning.loggers import TensorBoardLogger

	pl.seed_everything(args.seed)

	from ._data import LineageDataModule
	from .preprocess import NUM_FEATURES
	from ._model import LineageClassifierDeep, LineageClassifierLogistic
	from ._kfold import KFoldLoop

	datamodule = LineageDataModule(
		args.fp_data_train, args.fp_data_test,
		batch_size=args.batch_size, num_workers=12,
	)
	datamodule.setup('init')  # make dataset attr available, so we can compute class weights

	config = dict(
		# constants
		input_dim=NUM_FEATURES,  # see preprocess.py
		# model hyperparameters
		num_hidden=args.num_hidden,
		size_hidden=args.size_hidden,
		dropout=args.dropout,
		# training parameters
		lr=args.lr,
		lr_decay=args.lr_decay,
		seed=args.seed,
		num_epochs=args.num_epochs,
		batch_size=args.batch_size,
		# dataset parameters
		# fp_data_train=str(args.fp_data_train),
		num_folds=args.num_folds,
		class_weights=datamodule.train_dataset.class_weights().tolist(),
		# equalize=args.equalize,
	)
	if args.model_str == 'deep':
		model = LineageClassifierDeep(config)
	elif args.model_str == 'logistic':
		model = LineageClassifierLogistic(config)

	logger.info(model)
	logger.info(f'train/val dataset len={len(datamodule.train_dataset)}, test dataset len={len(datamodule.test_dataset)}')

	from hashlib import md5
	import os.path
	model_name = ' '.join(f'{key}={value}' for key, value in config.items())
	# model_name = md5(' '.join(f'{key}={value}' for key, value in config.items()).encode()).hexdigest()

	tblogger = TensorBoardLogger(
		save_dir=args.dir_logs,
		name=model_name,
		default_hp_metric=False
	)

	trainer = pl.Trainer(
		default_root_dir=args.dir_logs,
		max_epochs=args.num_epochs,
		accelerator='gpu', devices=1,
		enable_model_summary=True,
		num_sanity_val_steps=0,  # disable as this messes with kfold validation
		logger=tblogger,
		log_every_n_steps=1,
		check_val_every_n_epoch=1,
		callbacks=[
			ModelCheckpoint(
				os.path.join(tblogger.log_dir, 'kfolds_min_val_loss'),
				filename='model_{step}_{val_loss:.2f}',
				monitor='val_loss',
			)
			# EarlyStopping('val_loss', patience=20, mode='min'),  # early stopping messes with kfold validation
		],
		# TESTING
		# limit_train_batches=2,
		# limit_test_batches=2,
		# limit_val_batches=2,
		# 
	)

	if args.num_folds != 0:
		logger.info('Using k-fold validation')
		internal_fit_loop = trainer.fit_loop
		trainer.fit_loop = KFoldLoop(
			args.num_folds,
			export_path=os.path.join(tblogger.log_dir, 'kfolds')
		)
		trainer.fit_loop.connect(internal_fit_loop)
	else:
		logger.info('Disabled k-fold validation, no validation will be performed')
		datamodule.setup_folds(0)
		datamodule.setup_fold_index(0)

	trainer.fit(model, datamodule)

	if args.num_folds == 0:
		# run the test loop manually. ``KFoldLoop`` runs it automatically
		trainer.test(model, datamodule)