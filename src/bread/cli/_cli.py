import argparse
import pathlib
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('bread.cli')
import bread.algo.lineage

__all__ = ['main']

def main():
	parser = argparse.ArgumentParser(
		prog='bread.cli',
		description='Lineage guesser for budding yeast cells.',
		formatter_class=argparse.ArgumentDefaultsHelpFormatter
	)

	parser.add_argument(
		'--output-file',
		help='Path to output file (csv format)',
		type=pathlib.Path,
		required=True
	)

	parser.add_argument(
		'--fov',
		help='Field of view to process (integer)',
		type= int,
		required=False,
		default = 0
	)

	subparsers_lineage = parser.add_subparsers(
		dest='lineage_algo',
		required=True,
	)

	def add_majority_vote_doc(parser):
		parser.add_argument(
			'--num_frames',
			help='Number of frames to make guesses for after the bud has appeared. The algorithm makes a guess for each frame, then predicts a parent by majority-vote policy',
			default=bread.algo.lineage._lineage._MajorityVoteMixin.num_frames
		)
		parser.add_argument(
			'--offset_frames',
			help='Wait this number of frames after bud appears before guessing parent',
			default=bread.algo.lineage._lineage._MajorityVoteMixin.offset_frames
		)

	def add_guesser_doc(parser):
		parser.add_argument(
			'--segmentation-file',
			help='Path to segmentation file. Supported file types : hd5 (as exported by YeaZ)',
			type=pathlib.Path,
			required=True
		)
		parser.add_argument(
			'--nn_threshold',
			help='Cell masks separated by less than this threshold are considered neighbours',
			default=bread.algo.lineage._lineage.LineageGuesser.nn_threshold,
			type=float,
		)
		parser.add_argument(
			'--flexible_nn_threshold',
			help='If no nearest neighbours are found within the given threshold, try to find the closest one',
			default=bread.algo.lineage._lineage.LineageGuesser.flexible_nn_threshold,
			type=bool,
		)
		parser.add_argument(
			'--num_frames_refractory',
			help='After a parent cell has budded, exclude it from the parent pool in the next frames. It is recommended to set it to a low estimate, as high values will cause mistakes to propagate in time. A value of 0 corresponds to no refractory period.',
			default=bread.algo.lineage._lineage.LineageGuesser.num_frames_refractory,
			type=int,
		)


	# Budneck

	parser_budneck = subparsers_lineage.add_parser(
		'budneck',
		help='Guess lineage relations by looking at the budneck marker intensity along the contour of the bud.',
		formatter_class=argparse.ArgumentDefaultsHelpFormatter
	)
	parser_budneck.add_argument(
		'--budneck-file',
		help='Path to file containing budneck marker. Supported file types : tiff (as exported by Fiji)',
		type=pathlib.Path,
		required=True
	)
	add_guesser_doc(parser_budneck)
	add_majority_vote_doc(parser_budneck)
	parser_budneck.add_argument(
		'--kernel_N',
		help='Size of the gaussian smoothing kernel in pixels. larger means smoother intensity curves',
		default=bread.algo.lineage._lineage.LineageGuesserBudLum.kernel_N,
		type=int,
	)
	parser_budneck.add_argument(
		'--kernel_sigma',
		help='Number of standard deviations to consider for the smoothing kernel',
		default=bread.algo.lineage._lineage.LineageGuesserBudLum.kernel_sigma,
		type=int,
	)

	# ML

	parser_ml = subparsers_lineage.add_parser(
		'ml',
		help='Guess lineage relations by maximizing the expansion velocity of the bud with respect to the candidate parent.',
		formatter_class=argparse.ArgumentDefaultsHelpFormatter
	)
	add_guesser_doc(parser_ml)
	parser_ml.add_argument(
		'--num_frames',
		help='How many frames to consider to compute expansion velocity. At least 2 frames should be considered for good results',
		default=bread.algo.lineage._lineage.LineageGuesserExpansionSpeed.num_frames,
		type=int
	)
	parser_ml.add_argument(
		'--bud_distance_max',
		help='Maximal distance (in pixels) between points on the parent and bud contours to be considered as part of the "budding interface"',
		default=bread.algo.lineage._lineage.LineageGuesserExpansionSpeed.bud_distance_max,
		type=int
	)
	
	# Expansion speed

	parser_expspeed = subparsers_lineage.add_parser(
		'expspeed',
		help='Guess lineage relations by maximizing the expansion velocity of the bud with respect to the candidate parent.',
		formatter_class=argparse.ArgumentDefaultsHelpFormatter
	)
	add_guesser_doc(parser_expspeed)
	parser_expspeed.add_argument(
		'--num_frames',
		help='How many frames to consider to compute expansion velocity. At least 2 frames should be considered for good results',
		default=bread.algo.lineage._lineage.LineageGuesserExpansionSpeed.num_frames,
		type=int
	)
	parser_expspeed.add_argument(
		'--ignore_dist_nan',
		help='In some cases the computed expansion distance encounters an error (candidate parent flushed away, invalid contour, etc.), then the computed distance is replaced by nan for the given frame. If this happens for many frames, the computed expansion speed might be nan. Enabling this parameter ignores candidates for which the computed expansion speed is nan, otherwise raises an error.',
		default=bread.algo.lineage._lineage.LineageGuesserExpansionSpeed.ignore_dist_nan,
		type=bool
	)
	parser_expspeed.add_argument(
		'--bud_distance_max',
		help='Maximal distance (in pixels) between points on the parent and bud contours to be considered as part of the "budding interface"',
		default=bread.algo.lineage._lineage.LineageGuesserExpansionSpeed.bud_distance_max,
		type=int
	)

	# MinTheta

	parser_mintheta = subparsers_lineage.add_parser(
		'mintheta',
		help='Guess lineage relations by minimizing the angle between the major axis of the candidates and candidate-to-bud vector.',
		formatter_class=argparse.ArgumentDefaultsHelpFormatter
	)
	add_guesser_doc(parser_mintheta)
	add_majority_vote_doc(parser_mintheta)

	# MinDist

	parser_mindist = subparsers_lineage.add_parser(
		'mindist',
		help='Guess lineage relations by finding the cell closest to the bud, when it appears on the segmentation.',
		formatter_class=argparse.ArgumentDefaultsHelpFormatter
	)
	add_guesser_doc(parser_mindist)

	# Do stuff

	args = parser.parse_args()
	logger.debug(args)

	if args.lineage_algo == 'budneck':
		from bread.algo.lineage import LineageGuesserBudLum
		from bread.data import Segmentation, Lineage, Microscopy, SegmentationFile

		logger.info('Loading segmentation...')
		segmentationFile = SegmentationFile.from_h5(args.segmentation_file)
		fov = args.fov
		segmentation = segmentationFile.get_segmentation("FOV"+str(fov))
		logger.info(f'Loaded segmentation {segmentation}')

		logger.info('Loading budneck channel movie...')
		budneck_img = Microscopy.from_tiff(args.budneck_file)
		budneck_img = budneck_img.get_fov('FOV'+str(fov))
		logger.info(f'Loaded budneck channel movie {budneck_img}')

		logger.info('Loading guesser...')
		guesser = LineageGuesserBudLum(
			segmentation=segmentation,
			budneck_img=budneck_img,
			nn_threshold=args.nn_threshold,
			flexible_nn_threshold=args.flexible_nn_threshold,
			num_frames_refractory=args.num_frames_refractory,
			num_frames=args.num_frames,
			offset_frames=args.offset_frames,
			kernel_N=args.kernel_N,
			kernel_sigma=args.kernel_sigma,
		)
		logger.info(f'Loaded guesser {guesser}')

		logger.info(f'Running guesser...')
		lineage_guess: Lineage = guesser.guess_lineage()
		
		logger.info(f'Saving lineage...')
		lineage_guess.save_csv(args.output_file)

	if args.lineage_algo == 'ml':
		from bread.algo.lineage import LineageGuesserML
		from bread.data import Segmentation, Lineage, SegmentationFile

		logger.info('Loading segmentation...')
		segmentationFile = SegmentationFile.from_h5(args.segmentation_file)
		fov = args.fov
		segmentation = segmentationFile.get_segmentation("FOV"+str(fov))
		logger.info(f'Loaded segmentation {segmentation}')

		logger.info('Loading guesser...')
		guesser = LineageGuesserML(
			segmentation=segmentation,
			nn_threshold=args.nn_threshold,
			flexible_nn_threshold=args.flexible_nn_threshold,
			num_frames_refractory=args.num_frames_refractory,
			num_frames=args.num_frames,
			bud_distance_max=args.bud_distance_max
		)
		logger.info(f'Loaded guesser {guesser}')

		logger.info(f'Running guesser...')
		lineage_guess: Lineage = guesser.guess_lineage()
		
		logger.info(f'Saving lineage...')
		lineage_guess.save_csv(args.output_file)


	if args.lineage_algo == 'expspeed':
		from bread.algo.lineage import LineageGuesserExpansionSpeed
		from bread.data import Segmentation, Lineage, SegmentationFile

		logger.info('Loading segmentation...')
		segmentationFile = SegmentationFile.from_h5(args.segmentation_file)
		fov = args.fov
		segmentation = segmentationFile.get_segmentation("FOV"+str(fov))

		logger.info(f'Loaded segmentation {segmentation}')

		logger.info('Loading guesser...')
		guesser = LineageGuesserExpansionSpeed(
			segmentation=segmentation,
			nn_threshold=args.nn_threshold,
			flexible_nn_threshold=args.flexible_nn_threshold,
			num_frames_refractory=args.num_frames_refractory,
			num_frames=args.num_frames,
			ignore_dist_nan=args.ignore_dist_nan,
			bud_distance_max=args.bud_distance_max
		)
		logger.info(f'Loaded guesser {guesser}')

		logger.info(f'Running guesser...')
		lineage_guess: Lineage = guesser.guess_lineage()
		
		logger.info(f'Saving lineage...')
		lineage_guess.save_csv(args.output_file)

	if args.lineage_algo == 'mintheta':
		from bread.algo.lineage import LineageGuesserMinTheta
		from bread.data import Segmentation, Lineage, SegmentationFil

		logger.info('Loading segmentation...')
		segmentationFile = SegmentationFile.from_h5(args.segmentation_file)
		fov = args.fov
		segmentation = segmentationFile.get_segmentation("FOV"+str(fov))

		logger.info(f'Loaded segmentation {segmentation}')

		logger.info('Loading guesser...')
		guesser = LineageGuesserMinTheta(
			segmentation=segmentation,
			nn_threshold=args.nn_threshold,
			flexible_nn_threshold=args.flexible_nn_threshold,
			num_frames_refractory=args.num_frames_refractory,
			num_frames=args.num_frames,
			offset_frames=args.offset_frames,
		)
		logger.info(f'Loaded guesser {guesser}')

		logger.info(f'Running guesser...')
		lineage_guess: Lineage = guesser.guess_lineage()
		
		logger.info(f'Saving lineage...')
		lineage_guess.save_csv(args.output_file)

	if args.lineage_algo == 'mindist':
		from bread.algo.lineage import LineageGuesserMinDistance
		from bread.data import Segmentation, Lineage, SegmentationFile

		logger.info('Loading segmentation...')
		segmentationFile = SegmentationFile.from_h5(args.segmentation_file)
		fov = args.fov
		segmentation = segmentationFile.get_segmentation("FOV"+str(fov))

		logger.info(f'Loaded segmentation {segmentation}')

		logger.info('Loading guesser...')
		guesser = LineageGuesserMinDistance(
			segmentation=segmentation,
			nn_threshold=args.nn_threshold,
			flexible_nn_threshold=args.flexible_nn_threshold,
			num_frames_refractory=args.num_frames_refractory,
		)
		logger.info(f'Loaded guesser {guesser}')

		logger.info(f'Running guesser...')
		lineage_guess: Lineage = guesser.guess_lineage()
		
		logger.info(f'Saving lineage...')
		lineage_guess.save_csv(args.output_file)