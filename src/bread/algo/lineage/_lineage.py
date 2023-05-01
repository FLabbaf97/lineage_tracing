from abc import ABC, abstractmethod
from typing import Callable, Optional, List, Tuple
from dataclasses import dataclass, field
import warnings
import numpy as np
import scipy.signal
import scipy.spatial.distance
import scipy.ndimage

from bread.data import Lineage, Microscopy, Segmentation, Ellipse, Contour, BreadException, BreadWarning, Features

__all__ = [
	'LineageGuesser', 'LineageGuesserBudLum', 'LineageGuesserExpansionSpeed', 'LineageGuesserMinDistance', 'LineageGuesserMinTheta', 'LineageGuesserML',
	'LineageException', 'LineageWarning',
	'NotEnoughFramesException', 'NotEnoughFramesWarning'
]

class LineageException(BreadException):
	pass

class LineageWarning(BreadWarning):
	pass

class NotEnoughFramesException(LineageException):
	def __init__(self, bud_id: int, time_id: int, num_requested: int, num_remaining: int):
		super().__init__(f'Not enough frames in the movie (bud #{bud_id} at frame #{time_id}), requested {num_requested}, but only {num_remaining} remaining (need at least 2).')

class NotEnoughFramesWarning(LineageWarning):
	def __init__(self, bud_id: int, time_id: int, num_requested: int, num_remaining: int):
		super().__init__(f'Not enough frames in the movie (bud #{bud_id} at frame #{time_id}), requested {num_requested}, but only {num_remaining} remaining.')

@dataclass
class LineageGuesser(ABC):
	"""Construct LineageGuesser

	Parameters
	----------
	seg : Segmentation
	nn_threshold : float, optional
		Cell masks separated by less than this threshold are considered neighbours. by default 8.0.
	flexible_threshold : bool, optional
		If no nearest neighbours are found within the given threshold, try to find the closest one, by default False.
	num_frames_refractory : int, optional
		After a parent cell has budded, exclude it from the parent pool in the next frames.
		It is recommended to set it to a low estimate, as high values will cause mistakes to propagate in time.
		A value of 0 corresponds to no refractory period.
		by default 0.
	scale_length : float, optional
		Units of the segmentation pixels, in [length unit]/px, used for feature extraction (passed to ``Features``).
		by default 1.
	scale_time : float, optional
		Units of the segmentation frame separation, in [time unit]/frame, used for feature extraction (passed to ``Features``).
		by default 1.
	"""

	segmentation: Segmentation
	nn_threshold: float = 8
	flexible_nn_threshold: bool = False
	num_frames_refractory: int = 0
	scale_length: float = 1  # [length unit]/px
	scale_time: float = 1  # [time unit]/frame
	
	_cellids_refractory: dict = field(init=False, repr=False, default_factory=dict)
	_features: Features = field(init=False, repr=False)

	class NoGuessWarning(LineageWarning):
		def __init__(self, bud_id: int, time_id: int, error: Exception):
			super().__init__(f'Unable to determine parent for bud #{bud_id} in frame #{time_id}. Got error {repr(error)}')

	class NoCandidateParentException(LineageException):
		def __init__(self, time_id: int):
			super().__init__(f'No candidate parents have been found for in frame #{time_id}.')

	def __post_init__(self):
		self._features = Features(
			segmentation=self.segmentation,
			scale_length=self.scale_length, scale_time=self.scale_time,
			nn_threshold=self.nn_threshold
		)

	@abstractmethod
	def guess_parent(self, bud_id: int, time_id: int) -> int:
		"""Guess the parent associated to a bud

		Parameters
		----------
		bud_id : int
			id of the bud in the segmentation
		time_id : int
			frame index in the movie

		Returns
		-------
		parent_id : int
			guessed parent id
		"""
		raise NotImplementedError()

	def guess_lineage(self, progress_callback: Optional[Callable[[int, int], None]] = None):
		"""Guess the full lineage of a given bud.

		Returns
		-------
		lineage: Lineage
			guessed lineage
		progress_callback: Callable[[int, int]] or None
			callback for progress
		"""

		lineage_init: Lineage = self.segmentation.find_buds()
		bud_ids, time_ids = lineage_init.bud_ids, lineage_init.time_ids
		parent_ids = np.empty_like(bud_ids, dtype=int)
		# bud_ids, time_ids = bud_ids[time_ids > 0], time_ids[time_ids > 0]

		for i, (bud_id, time_id) in enumerate(zip(bud_ids, time_ids)):
			if progress_callback is not None:
				progress_callback(i, len(time_ids))
			if time_id == 0:
				# cells in first frame have no parent
				parent_ids[i] = Lineage.SpecialParentIDs.PARENT_OF_ROOT.value
				continue

			try:
				parent_ids[i] = self.guess_parent(int(bud_id), int(time_id))  # BUGFIX : cast to an int because indexing with numpy.uint64 raises an error
				self._cellids_refractory[parent_ids[i]] = self.num_frames_refractory
			except BreadException as e:
				if isinstance(e, LineageGuesser.NoCandidateParentException):
					# the cell is too far away from any other cells, it does not belong to the colony
					parent_ids[i] = Lineage.SpecialParentIDs.PARENT_OF_EXTERNAL.value
				else:
					# the guesser could not give a guess
					warnings.warn(LineageGuesser.NoGuessWarning(bud_id, time_id, e))
					parent_ids[i] = Lineage.SpecialParentIDs.NO_GUESS.value

			self._decrement_refractory()

		return Lineage(parent_ids, bud_ids, time_ids)

	def _decrement_refractory(self):
		"""Decrement the time remaining for refractory cells"""

		pop_cell_ids = []

		# decrement cell timers
		for cell_id in self._cellids_refractory.keys():
			self._cellids_refractory[cell_id] -= 1
			if self._cellids_refractory[cell_id] < 0:
				pop_cell_ids.append(cell_id)

		# remove cells which have finished their refractory period
		for cell_id in pop_cell_ids:
			self._cellids_refractory.pop(cell_id)

	def _candidate_parents(self, time_id: int, excluded_ids: Optional[List[int]] = None, nearest_neighbours_of: Optional[int] = None) -> np.ndarray:
		"""Generate a list of candidate parents to consider for budding events.

		Parameters
		----------
		time_id : int
			frame index in the movie.
		excluded_ids : list[int], optional
			Exclude these cell ids from the candidates.
		nearest_neighbours_of : int or None
			Exclude cells which are not nearest neighbours of the cell with id ``nearest_neighbours_of``.
			Cells for which the smallest distance to cell ``nearest_neighbours_of`` is less than ``self._nn_threshold`` are considered nearest neighbours.
			default is `None`.

		Returns
		-------
		candidate_ids : array-like of int
			ids of the candidate parents
		"""

		cell_ids_prev, cell_ids_curr = self.segmentation.cell_ids(time_id-1), self.segmentation.cell_ids(time_id)
		bud_ids = np.setdiff1d(cell_ids_curr, cell_ids_prev)
		candidate_ids = np.setdiff1d(cell_ids_curr, bud_ids)

		# remove the excluded_ids
		if excluded_ids is not None:
			candidate_ids = np.setdiff1d(candidate_ids, excluded_ids)

		# remove refractory cells
		refractory_ids = np.array(list(self._cellids_refractory.keys()))
		candidate_ids = np.setdiff1d(candidate_ids, refractory_ids)

		# nearest neighbours
		if nearest_neighbours_of is not None:
			# we don't use self._features._nearest_neighour_of, because we need the min dists fallback for the flexible_nn_threshold
			# however we exploit the caching capabilities of Features !

			contour_bud = self._features._contour(nearest_neighbours_of, time_id)
			dists = np.zeros_like(candidate_ids, dtype=float)

			for i, parent_id in enumerate(candidate_ids):
				contour_parent = self._features._contour(parent_id, time_id)
				dists[i] = self._features._nearest_points(contour_bud, contour_parent)[-1]

			if any(dists <= self.nn_threshold):
				# the cell has nearest neighbours
				candidate_ids = candidate_ids[dists <= self.nn_threshold]
			else:
				# the cell has no nearest neighbours
				if self.flexible_nn_threshold:
					# pick the closest cell
					candidate_ids = candidate_ids[[np.argmin(dists)]]
				else:
					# warn that the cell has no nearest neighbours
					warnings.warn(BreadWarning(f'cell #{nearest_neighbours_of} does not have nearest neighbours with a distance less than {self.nn_threshold}, and flexible_threshold is {self.flexible_nn_threshold}.'))
					candidate_ids = np.array(tuple())

		if len(candidate_ids) == 0:
			# no nearest neighbours
			raise LineageGuesser.NoCandidateParentException(time_id)

		return candidate_ids


@dataclass
class _BudneckMixin:
	# dataclass fields without default value cannot appear after data fields with default values
	# this class provides a mixin to add the positional budneck_img argument
	budneck_img: Microscopy


@dataclass
class _MajorityVoteMixin:
	"""Redefine ``guess_parent`` to use a majority vote. The method uses an abstract method ``_guess_parent_singleframe`` that needs to be implemented by subclasses"""

	num_frames: int = 5
	offset_frames: int = 0

	def __post_init__(self):
		assert self.num_frames > 0, f'num_frames must be strictly greater than zero, got num_frames={self.num_frames}'

	def guess_parent(self, bud_id: int, time_id: int) -> int:		
		"""Guess the parent associated to a bud

		Parameters
		----------
		bud_id : int
			id of the bud in the segmentation
		time_id : int
			frame index in the movie

		Returns
		-------
		parent_id : int
			guessed parent id
		"""

		frame_range = self.segmentation.request_frame_range(time_id + self.offset_frames, time_id + self.offset_frames + self.num_frames)

		if len(frame_range) == 0:
			raise NotEnoughFramesException(bud_id, time_id, self.num_frames, len(frame_range))
		if len(frame_range) < self.num_frames:
			warnings.warn(NotEnoughFramesWarning(bud_id, time_id, self.num_frames, len(frame_range)))

		parent_ids = []
		exceptions = []

		# guess a parent for each frame in [t + offset, t + offset + duration)
		for time_id_ in frame_range:
			try:
				parent_ids.append(self._guess_parent_singleframe(bud_id, time_id_))
			except BreadException as e:
				# raise e
				exceptions.append(e)
				warnings.warn(LineageWarning(f'Unable to determine parent, got exception {repr(e)}. Computation for frame #{time_id_}, bud #{bud_id} skipped.'))

		if len(parent_ids) == 0:
			# test if the exceptions were raised due to no parent candidates
			if all(isinstance(e, LineageGuesser.NoCandidateParentException) for e in exceptions):
				return Lineage.SpecialParentIDs.PARENT_OF_EXTERNAL.value

			raise LineageException(f'All of the frames studied ({list(frame_range)}) for bud #{bud_id} (at frame #{time_id}) gave an exception.')

		# perform majority vote
		values, counts = np.unique(parent_ids, return_counts=True)
		majority_ids = values[(counts == np.max(counts))]

		# if vote is ambiguous, i.e. 2 or more parents have been guessed the maximum number of times
		# then the nearest parent is returned
		if len(majority_ids) > 1:
			contour_bud = self._features._contour(bud_id, time_id)
			dists = np.zeros_like(majority_ids, dtype=float)

			# BUG : this needs to check for `flexible_nn`, eg colony004 bud=113
			for i, parent_id in enumerate(majority_ids):
				contour_parent = self._features._contour(parent_id, time_id)
				dists[i] = self._features._nearest_points(contour_bud, contour_parent)[-1]

			return majority_ids[np.argmin(dists)]
			# warnings.warn(BreadWarning(f'Ambiguous vote for frame #{time_id}, bud #{bud_id}. Possible parents : {majority_ids}'))

		return majority_ids[0]

	@abstractmethod
	def _guess_parent_singleframe(self, bud_id, time_id):
		pass


@dataclass
class LineageGuesserBudLum(_MajorityVoteMixin, LineageGuesser, _BudneckMixin):
	"""Guess lineage relations by looking at the budneck marker intensity along the contour of the bud.

	Parameters
	----------
	segmentation : Segmentation
	budneck_img : Microscopy
	nn_threshold : float, optional
		Cell masks separated by less than this threshold are considered neighbors, by default 8.0.
	flexible_nn_threshold : bool, optional
		If no nearest neighbours are found within the given threshold, try to find the closest one, by default False.
	num_frames_refractory : int, optional
		After a parent cell has budded, exclude it from the parent pool in the next frames.
		It is recommended to set it to a low estimate, as high values will cause mistakes to propagate in time.
		A value of 0 corresponds to no refractory period.
		by default 0.
	num_frames : int, default 5
		Number of frames to watch the budneck marker channel for.
		The algorithm makes a guess for each frame, then predicts a parent by majority-vote policy.
	offset_frames : int, default 0
		Wait this number of frames after bud appears before guessing parent.
		Useful if the GFP peak is often delayed.
	kernel_N : int, default 30
		Size of the gaussian smoothing kernel in pixels. larger means smoother intensity curves.
	kernel_sigma : int, default 1
		Number of standard deviations to consider for the smoothing kernel.
	"""

	# budneck_img: Microscopy  # see _BudneckMixin
	# num_frames: int = 5  # see _MajorityVoteMixin
	# offset_frames: int = 0
	kernel_N: int = 30
	kernel_sigma: int = 1

	def __post_init__(self):
		LineageGuesser.__post_init__(self)
		assert self.offset_frames >= 0, f'offset_frames must be greater or equal to zero, got offset_frames={self.offset_frames}'
		assert self.segmentation.data.shape == self.budneck_img.data.shape, f'segmentation and budneck imaging must have the same shape, got segmentation data of shape {self.segmentation.data.shape} and budneck marker data of shape {self.budneck_img.data.shape}'
		
	def _guess_parent_singleframe(self, bud_id: int, time_id: int) -> int:
		"""Guess the parent of a bud, using the budneck marker at a certain frame.

		Parameters
		----------
		bud_id : int
			id of the bud in the segmentation
		time_id : int
			frame index in the movie

		Returns
		-------
		parent_id : int
			guessed parent id
		"""

		# explicitly exclude the bud from the considered candidates, because we also study frames after the bud was detected
		candidate_ids = self._candidate_parents(time_id, excluded_ids=[bud_id], nearest_neighbours_of=bud_id)
		contours = [self._features._contour(candidate_id, time_id) for candidate_id in candidate_ids]
		
		# Compute position of the GFP peak around the contour of the bud
		contour_peak = self._peak_contour_luminosity(bud_id, time_id)
		# Find the min distances between `contour_peak` and all other cell borders
		min_dists = [scipy.spatial.distance.cdist(contour.data, contour_peak[None, :]).min() for contour in contours]

		# Find index of cell closest to the YFP peak -> parent id
		i_min = min_dists.index(min(min_dists))
		parent_id = candidate_ids[i_min]

		return parent_id

	def _contour_luminosity(self, bud_id: int, time_id: int):
		"""Return contour of a cell mask and the corresponding averaged intensity of a marker along the border.

		Parameters
		----------
		bud_id : int
			id of the bud in the segmentation
		time_id : int
			frame index in the movie

		Returns
		-------
		contour : Contour
			contour of the bud
		luminosities : numpy.ndarray (shape=(N,), dtype=float)
			averaged luminosities in each of the countour points
		"""

		contour = self._features._contour(bud_id, time_id)

		t = np.linspace(-3, 3, self.kernel_N, dtype=np.float64)
		bump = np.exp(-0.5*(t/self.kernel_sigma)**2)
		bump /= np.trapz(bump)  # normalize the integral to 1
		kernel = bump[:, np.newaxis] * bump[np.newaxis, :]  # make a 2D kernel

		# Convolve only a small bounding box

		# Find the cell bounding box
		ymin, ymax, xmin, xmax = contour[:, 1].min(), contour[:, 1].max(), contour[:, 0].min(), contour[:, 0].max()
		# Add enough space to fit the convolution kernel on the border
		ymin, ymax, xmin, xmax = ymin-self.kernel_N, ymax+self.kernel_N, xmin-self.kernel_N, xmax+self.kernel_N
		# Clamp indices to prevent out of bound errors (shape[0]-1 instead of shape because ymax (resp. xmax) is inclusive)
		ymin, ymax = np.clip((ymin, ymax), 0, self.budneck_img[time_id].shape[0]-1)
		xmin, xmax = np.clip((xmin, xmax), 0, self.budneck_img[time_id].shape[1]-1)
		# Recover trimmed image
		img_trimmed = self.budneck_img[time_id, ymin:ymax+1, xmin:xmax+1]

		# Perform smoothing

		# Using the scipy fftconvolution
		# mode='same' is there to enforce the same output shape as input arrays (ie avoid border effects)
		convolved = scipy.signal.fftconvolve(img_trimmed, kernel, mode='same')
		luminosities = convolved[contour[:, 1]-ymin, contour[:, 0]-xmin]

		return contour, luminosities

	def _peak_contour_luminosity(self, bud_id: int, time_id: int) -> np.ndarray:
		"""Return pixel index (i, j) of the maximum luminosity around the border.

		Parameters
		----------
		bud_id : int
			id of the bud in the segmentation
		time_id : int
			frame index in the movie

		Returns
		-------
		contour_peak : numpy.ndarray (shape=(2,))
			pixel index of the maximum luminosity
		"""

		contours, luminosities = self._contour_luminosity(bud_id, time_id)

		i_peak = np.argmax(luminosities)
		contour_peak = contours[i_peak]

		return contour_peak


@dataclass
class LineageGuesserML(LineageGuesser):
	"""Guess lineage relations using a machine learning model with multiple input features that generated using segmentation file.
	Parameters
	----------
	segmentation : Segmentation
	nn_threshold : float, optional
		cell masks separated by less than this threshold are considered neighbors, by default 8.0.
	flexible_nn_threshold : bool, optional
		If no nearest neighbours are found within the given threshold, try to find the closest one, by default False.
	num_frames_refractory : int, optional
		After a parent cell has budded, exclude it from the parent pool in the next frames.
		It is recommended to set it to a low estimate, as high values will cause mistakes to propagate in time.
		A value of 0 corresponds to no refractory period.
		by default 0.
	num_frames : int, optional
		How many frames to consider to compute expansion velocity.
		At least 2 frames should be considered for good results.
		by default 5.
	bud_distance_max : float, optional
		Maximal distance (in pixels) between points on the parent and bud contours to be considered as part of the "budding interface".
		by default 7.
	"""
	num_frames: int = 5
	bud_distance_max: float = 8
	num_nn_threshold: int = 4
	number_of_features: int = 10

	def __post_init__(self):
		LineageGuesser.__post_init__(self)
		assert self.num_frames >= 2, f'not enough consecutive frames considered for analysis, got {self.num_frames}.'
		self._features.budding_time = self.num_frames
		self._features.bud_distance_max = self.bud_distance_max
		import xgboost as xgb

		# Load the saved model
		# TODO: fix this type of path, make it relative to package
		self.model_path = '/home/farzaneh/Documents/Bread/bread/src/bread/algo/lineage/XGBoost/best_model_for_matrix_data_10_features.json'
		self.model = xgb.Booster()
		self.model.load_model(self.model_path)

		# set values for model inputes
		self.num_nn_threshold = 4
		self.number_of_features = 10
	
	def _get_ml_features(self, bud_id, candidate_id, time_id, selected_times):
		"""Return the features to be used by the ML model for a given bud and candidate at a given time.
		Parameters
		----------
		bud_id : int
			id of the bud in the segmentation
		candidate_id : int
			id of the candidate in the segmentation
		time_id : int
			frame index in the movie
		Returns
		-------
		summary features : numpy.ndarray (shape=(N,), dtype=float),
			features to be used by the ML model
		full features : numpy.ndarray (shape=(M,), dtype=float),
			full set of features related to the bud and candidate
		"""
		budcm_to_budpt_l = np.zeros(self.num_frames, dtype=np.float64)
		budcm_to_candidatecm_l = np.zeros(self.num_frames, dtype=np.float64)
		expansion_vector_l = np.zeros(self.num_frames, dtype=np.float64)
		position_bud = np.zeros(self.num_frames, dtype=np.float64)
		orientation_bud = np.zeros(self.num_frames, dtype=np.float64)
		distances = np.zeros(self.num_frames, dtype=np.float64)
		for i_t, t in enumerate(selected_times):
			candidatemaj = self._features.cell_maj(candidate_id, t)
			budmaj = self._features.cell_maj(bud_id, t)
			budcm_to_budpt = self._features.pair_budcm_to_budpt(t, bud_id, candidate_id)
			budcm_to_candidatecm = self._features.pair_cmtocm(t, bud_id, candidate_id)
			candidatecm_to_budpt = budcm_to_budpt - budcm_to_candidatecm
			expansion_vector = self._features._expansion_vector(bud_id, candidate_id, t)
			budcm_to_candidatecm_l[i_t] = np.linalg.norm(budcm_to_candidatecm)
			budcm_to_budpt_l[i_t] = np.linalg.norm(budcm_to_budpt)
			expansion_vector_l[i_t] = np.linalg.norm(expansion_vector)

			# distance between bud and candidate
			distances[i_t] = self._features.pair_dist(t, bud_id, candidate_id)
			# position on candidate where bud appears
			# both matter, the position (preferential toward major axis) as well as the change in position (no movement allowed)
			innerproduct = np.dot(candidatemaj, candidatecm_to_budpt)
			position_bud[i_t] = np.arccos(np.absolute(innerproduct) / np.linalg.norm(candidatemaj) / np.linalg.norm(candidatecm_to_budpt))   

			# orientation of bud with respect to candidate
			innerproduct = np.dot(budmaj, candidatecm_to_budpt)
			orientation_bud[i_t] = np.arccos(np.absolute(innerproduct) / np.linalg.norm(budmaj) / np.linalg.norm(candidatecm_to_budpt))
		
		features_summary = []
		features_full = []

		# distances
		features_summary.append(distances[0])
		features_summary.append(distances.max())
		
		features_full.extend(distances)

		# growth
		m = np.polyfit(selected_times,budcm_to_candidatecm_l[0:len(selected_times)],1)[0]
		m_budpt = np.polyfit(selected_times,budcm_to_budpt_l[0:len(selected_times)],1)[0]
		m_exvec = np.polyfit(selected_times,expansion_vector_l[0:len(selected_times)],1)[0]
		features_summary.append(m)
		features_summary.append(m_budpt)
		features_summary.append(m_exvec)
		
		features_full.extend(budcm_to_candidatecm_l)
		
		features_full.extend(budcm_to_budpt_l)
		
		features_full.extend(expansion_vector_l)

		# position and movement around mother
		features_summary.append(position_bud[0])
		features_summary.append(position_bud.std())
		
		features_full.extend(position_bud)

		# orientation of bud
		features_summary.append(orientation_bud[0])

		m = np.polyfit(selected_times,orientation_bud[0:len(selected_times)],1)[0]
		features_summary.append(m)
		features_summary.append(orientation_bud[len(selected_times)-1])
		
		features_full.extend(orientation_bud)

		return features_summary, features_full
	
	def guess_parent(self, bud_id, time_id):
		candidate_parents = self._candidate_parents(time_id, nearest_neighbours_of=bud_id)
		frame_range = self.segmentation.request_frame_range(time_id, time_id + self.num_frames)
		num_frames_available = self.num_frames
		if len(frame_range) < 2:
			raise NotEnoughFramesException(bud_id, time_id, self.num_frames, len(frame_range))
		if len(frame_range) < self.num_frames:
			num_frames_available = len(frame_range)
			warnings.warn(NotEnoughFramesWarning(bud_id, time_id, self.num_frames, len(frame_range)))

		# check the bud still exists !
		for time_id_ in frame_range:
			if bud_id not in self.segmentation.cell_ids(time_id_):
				raise LineageGuesserExpansionSpeed.BudVanishedException(bud_id, time_id_)
		selected_times = [i for i in range(time_id, time_id + num_frames_available)]

		# get features for all candidates
		summary_features = np.zeros((len(candidate_parents), self.number_of_features), dtype=np.float64)
		for c_id , candidate in enumerate(candidate_parents):
			summary_features[c_id], _ = self._get_ml_features(bud_id, candidate, time_id, selected_times)
		
		# Find the id of parent with the highest probability using the ml model
		try:
			parent_id = self.predict_parent(bud_id, summary_features.reshape((1, ) + summary_features.shape))
			return candidate_parents[parent_id]
		except Exception as e:
			# check if there is only one candidate, if so return it
			if len(candidate_parents) == 1:
				return candidate_parents[0]
			# If the model fails, return -3 which means that there is no guess for this bud
			else:
				return -3
		

	def predict_parent(self, bud_id, batch_features):
		import itertools
		import xgboost as xgb

		if(self.model is None):
			raise Exception("No model was loaded")
		
		if batch_features.shape[1] <= self.num_nn_threshold:
			# Get the number of features in the model
			num_features = self.num_nn_threshold * self.number_of_features
			# Prepare the data for prediction
			X = batch_features #X is the set that is going to be used
			X = self._flatten_3d_array(X)
			if X.shape[1] < num_features:
				# Pad X with -1.0 for any missing features/candidates
				X_padded = np.pad(X, ((0, 0), (0, num_features - X.shape[1])), mode='constant', constant_values=-1.0)
			else:
				X_padded = X

			dtest = xgb.DMatrix(X_padded)
			# Make predictions using the loaded model
			preds = self.model.predict(dtest)
			preds = np.round(preds)
			return int(preds[0])
		
		# in case we have more candidates than the threshold, we need to test all combinations of 4 candidates
		elif batch_features.shape[1] > self.num_nn_threshold:
			# test for all combinations of 4 candidates and return the one with that repeats more times
			combinations = list(itertools.combinations(range(batch_features.shape[1]), 4))
			max_count = 0
			max_idx = 0
			pred_list = []
			for i, comb in enumerate(combinations):
				X = batch_features[:, comb]
				X = self._flatten_3d_array(X)
				# Pad X with -1.0 for any missing features/candidates
				num_features = self.num_nn_threshold * self.number_of_features
				if X.shape[1] < num_features:
					X_padded = np.pad(X, ((0, 0), (0, num_features - X.shape[1])), mode='constant', constant_values=-1.0)
				else:
					X_padded = X
				dtest = xgb.DMatrix(X_padded)
				preds = self.model.predict(dtest)
				pred = comb[int(preds[0])]
				pred_list.append(pred)
			
			from collections import Counter
			count = Counter(pred_list)
			most_common = count.most_common(1)
			return most_common[0][0] 

	def _keep_features(self, matrices, feature_columns = [0,1,2,3]):
		"""
		For a list of bud matrices keep only certain features
		"""
		new_matrices = np.zeros((matrices.shape[0], matrices.shape[1], len(feature_columns)))
		for i, fid in enumerate(feature_columns):
			new_matrices[:,:,i] = matrices[:,:,fid]
		return new_matrices

	def _flatten_3d_array(self,arr):
		"""
		Flattens a 3-dimensional numpy array while keeping the first dimension unchanged
		"""
		shape = arr.shape
		new_shape = (shape[0], np.prod(shape[1:]))
		return arr.reshape(new_shape)

		
@dataclass
class LineageGuesserExpansionSpeed(LineageGuesser):
	"""Guess lineage relations by maximizing the expansion velocity of the bud with respect to the candidate parent.
	
	Parameters
	----------
	segmentation : Segmentation
	nn_threshold : float, optional
		cell masks separated by less than this threshold are considered neighbors, by default 8.0.
	flexible_nn_threshold : bool, optional
		If no nearest neighbours are found within the given threshold, try to find the closest one, by default False.
	num_frames_refractory : int, optional
		After a parent cell has budded, exclude it from the parent pool in the next frames.
		It is recommended to set it to a low estimate, as high values will cause mistakes to propagate in time.
		A value of 0 corresponds to no refractory period.
		by default 0.
	num_frames : int, optional
		How many frames to consider to compute expansion velocity.
		At least 2 frames should be considered for good results.
		by default 5.
	ignore_dist_nan : bool, optional
		In some cases the computed expansion distance encounters an error (candidate parent flushed away, invalid contour, etc.),
		then the computed distance is replaced by nan for the given frame.
		If this happens for many frames, the computed expansion speed might be nan.
		Enabling this parameter ignores candidates for which the computed expansion speed is nan, otherwise raises an error.
		by default True.
	bud_distance_max : float, optional
		Maximal distance (in pixels) between points on the parent and bud contours to be considered as part of the "budding interface".
		by default 7.
	"""

	num_frames: int = 5
	ignore_dist_nan: bool = True
	bud_distance_max: float = 7

	class BudVanishedException(LineageException):
		def __init__(self, bud_id: int, time_id: int):
			super().__init__(f'Bud #{bud_id} vanished in between frame {time_id-1} to {time_id}.')

	class NanSpeedException(LineageException):
		def __init__(self, bud_id: int, parent_ids: int, time_id: int):
			super().__init__(f'Unable to determine parent for bud #{bud_id} at frame #{time_id}. The following parent candidates caused a problem : {parent_ids}')

	def __post_init__(self):
		LineageGuesser.__post_init__(self)
		assert self.num_frames >= 2, f'not enough consecutive frames considered for analysis, got {self.num_frames}.'
		self._features.budding_time = self.num_frames
		self._features.bud_distance_max = self.bud_distance_max

	def guess_parent(self, bud_id: int, time_id: int) -> int:
		"""Guess the parent associated to a bud

		Parameters
		----------
		bud_id : int
			id of the bud in the segmentation
		time_id : int
			frame index in the movie

		Returns
		-------
		parent_id : int
			guessed parent id
		"""

		candidate_parents = self._candidate_parents(time_id, nearest_neighbours_of=bud_id)
		frame_range = self.segmentation.request_frame_range(time_id, time_id + self.num_frames)

		if len(frame_range) < 2:
			raise NotEnoughFramesException(bud_id, time_id, self.num_frames, len(frame_range))
		if len(frame_range) < self.num_frames:
			warnings.warn(NotEnoughFramesWarning(bud_id, time_id, self.num_frames, len(frame_range)))

		# check the bud still exists !
		for time_id_ in frame_range:
			if bud_id not in self.segmentation.cell_ids(time_id_):
				raise LineageGuesserExpansionSpeed.BudVanishedException(bud_id, time_id_)

		dists_all = [self._features._expansion_distance(bud_id, parent_id, time_id) for parent_id in candidate_parents]
		# numpy.gradient can work with nan values
		# dists might contain nan values, we need to use numpy.nanmean to ignore them for the velocity computation
		mean_vels = np.array([np.nanmean(np.gradient(dists)) for dists in dists_all])

		# test for nan values which might have propagated to final velocities (not enough valid distances !)
		if not self.ignore_dist_nan and np.any(np.isnan(mean_vels)) > 0:
			raise LineageGuesserExpansionSpeed.NanSpeedException(bud_id, candidate_parents[np.isnan(mean_vels)], time_id)

		return candidate_parents[np.nanargmax(mean_vels)]


@dataclass
class LineageGuesserMinTheta(_MajorityVoteMixin, LineageGuesser):
	"""Guess lineage relations by minimizing the angle between the major axis of the candidates and candidate-to-bud vector.
	
	Parameters
	----------
	segmentation : Segmentation
	nn_threshold : float, optional
		cell masks separated by less than this threshold are considered neighbors, by default 8.0.
	flexible_nn_threshold : bool, optional
		If no nearest neighbours are found within the given threshold, try to find the closest one, by default False.
	num_frames_refractory : int, optional
		After a parent cell has budded, exclude it from the parent pool in the next frames.
		It is recommended to set it to a low estimate, as high values will cause mistakes to propagate in time.
		A value of 0 corresponds to no refractory period.
		by default 0.
	num_frames : int, default 5
		Number of frames to make guesses for after the bud has appeared.
		The algorithm makes a guess for each frame, then predicts a parent by majority-vote policy.
	offset_frames : int, default 0
		Wait this number of frames after bud appears before guessing parent.
	"""

	def __post_init__(self):
		LineageGuesser.__post_init__(self)

	def _guess_parent_singleframe(self, bud_id, time_id):
		candidate_ids = self._candidate_parents(time_id, excluded_ids=[bud_id], nearest_neighbours_of=bud_id)
		candidate_cms = [ self._features._cm(candidate_id, time_id) for candidate_id in candidate_ids ]
		bud_cm = self._features._cm(bud_id, time_id)
		ellipses = [self._features._ellipse(candidate_id, time_id) for candidate_id in candidate_ids]
		maj_axes = [np.array([np.sin(e.angle), np.cos(e.angle)]) for e in ellipses]  # WARNING : centers of mass are (y, x)
		vecs = [bud_cm - candidate_cm for candidate_cm in candidate_cms]
		nvecs = [vec / np.linalg.norm(vec) for vec in vecs]
		abscosthetas = [abs(np.dot(nvec, maj_ax)) for nvec, maj_ax in zip(nvecs, maj_axes)]
		return candidate_ids[abscosthetas.index(max(abscosthetas))]


@dataclass
class LineageGuesserMinDistance(LineageGuesser):
	"""Guess lineage relations by finding the cell closest to the bud, when it appears on the segmentation.
	
	Parameters
	----------
	segmentation : Segmentation
	nn_threshold : float, optional
		cell masks separated by less than this threshold are considered neighbors, by default 8.0.
	flexible_nn_threshold : bool, optional
		If no nearest neighbours are found within the given threshold, try to find the closest one, by default False.
	num_frames_refractory : int, optional
		After a parent cell has budded, exclude it from the parent pool in the next frames.
		It is recommended to set it to a low estimate, as high values will cause mistakes to propagate in time.
		A value of 0 corresponds to no refractory period.
		by default 0.
	"""

	def __post_init__(self):
		LineageGuesser.__post_init__(self)

	def guess_parent(self, bud_id: int, time_id: int) -> int:
		"""Guess the parent associated to a bud

		Parameters
		----------
		bud_id : int
			id of the bud in the segmentation
		time_id : int
			frame index in the movie

		Returns
		-------
		parent_id : int
			guessed parent id
		"""
		
		candidate_ids = self._candidate_parents(time_id, excluded_ids=[bud_id], nearest_neighbours_of=bud_id)

		if len(candidate_ids) == 0:
			return Lineage.SpecialParentIDs.PARENT_OF_EXTERNAL.value

		contour_bud = self._features._contour(bud_id, time_id)
		dists = np.zeros_like(candidate_ids, dtype=float)

		for i, parent_id in enumerate(candidate_ids):
			contour_parent = self._features._contour(parent_id, time_id)
			dists[i] = self._features._nearest_points(contour_bud, contour_parent)[-1]

		return candidate_ids[np.argmin(dists)]