from dataclasses import dataclass, field
from ._data import Segmentation, Contour, Ellipse, BreadWarning, BreadException
from typing import Any, Callable, List, Optional, Tuple, TypeVar
import scipy.spatial
import numpy as np
import warnings

__all__ = ['Features']

R = TypeVar('R')

@dataclass
class Features:
	"""Computes geometrical features from a segmentation

	Returns
	-------
	segmentation : Segmentation
	scale_length : float, optional
		Units of the segmentation pixels, in [length unit]/px, used for feature extraction (passed to ``Features``).
		by default 1.
	scale_time : float, optional
		Units of the segmentation frame separation, in [time unit]/frame, used for feature extraction (passed to ``Features``).
		by default 1.
	budding_time : int, optional
		Duration of budding event in frames, used for features which requires time-averaged quantities
		by default 5.
	nn_threshold : float, optional
		Cell masks separated by less than this threshold are considered neighbours.
		by default 8.0.
	bud_distance_max : float, optional
		Maximal distance (in pixels) between points on the parent and bud contours to be considered as part of the "budding interface".
		by default 7.0
	"""

	segmentation: Segmentation
	scale_length: float = 1  # [length unit]/px
	scale_time: float = 1  # [time unit]/frame
	budding_time: int = 5
	nn_threshold: float = 8.0
	bud_distance_max: float = 7.0

	_cached_contours: 'dict[Tuple[int, int], Contour]' = field(init=False, repr=False, default_factory=dict)
	_cached_ellipses: 'dict[Tuple[int, int], Ellipse]' = field(init=False, repr=False, default_factory=dict)
	_cached_ellipses_max_ecc: 'dict[Tuple[int, int, int], Tuple[Ellipse, int]]' = field(init=False, repr=False, default_factory=dict)
	_cached_cms: 'dict[Tuple[int, int], np.ndarray]' = field(init=False, repr=False, default_factory=dict)

	def cell_area(self, time_id: int, cell_id: int) -> float:
		"""area enclosed by the cell contour"""
		return self._contour(cell_id, time_id).area * self.scale_length**2
	
	def cell_r_equiv(self, time_id: int, cell_id: int) -> float:
		"""radius of the circle with same area as ellipse : r=sqrt(a*b)"""
		return self._ellipse(cell_id, time_id).r_equiv * self.scale_length
	
	def cell_r_maj(self, time_id: int, cell_id: int) -> float:
		"""major axis of the elliptical fit"""
		return self._ellipse(cell_id, time_id).r_maj * self.scale_length

	def cell_r_min(self, time_id: int, cell_id: int) -> float:
		"""minor axis of the elliptical fit"""
		return self._ellipse(cell_id, time_id).r_min * self.scale_length

	def cell_alpha(self, time_id: int, cell_id: int) -> float:
		"""angle of the major axis with the horizontal (1, 0) vector"""
		return self._ellipse(cell_id, time_id).angle

	def cell_ecc(self, time_id: int, cell_id: int) -> float:
		"""eccentricity of the cell"""
		return self._ellipse(cell_id, time_id).ecc

	def cell_maj(self, cell_id: int, time_id: int)-> float:
		cell_el = self._ellipse(cell_id, time_id)
		cell_maj = np.array([np.cos(cell_el.angle), np.sin(cell_el.angle)]) * cell_el.r_maj
		return cell_maj

	def cell_min(self, cell_id: int, time_id: int) -> float:
		"""Return the minor axis of the cell in the given time frame"""
		cell_el = self._ellipse(cell_id, time_id)
		cell_min = np.array([np.cos(cell_el.angle+np.pi/2), np.sin(cell_el.angle+np.pi/2)]) * cell_el.r_min
		return cell_min

	def pair_budcm_to_budpt(self, time_id, bud_id:int , candidate_id:int) -> np.ndarray:
		"""Return the vector from the bud center of mass to the bud point"""
		return self.pair_cmtocm(time_id, candidate_id, bud_id) + self.pair_budpt(time_id, candidate_id, bud_id) 

	def pair_cmtocm(self, time_id: int, cell_id1: int, cell_id2: int) -> np.ndarray:
		"""vector going from cm2 (center of mass) to cm1, in (x, y) coordinates"""
		return (self._cm(cell_id1, time_id) - self._cm(cell_id2, time_id))[[1, 0]] * self.scale_length

	def pair_dist(self, time_id: int, cell_id1: int, cell_id2: int) -> float:
		"""closest distance between the two membranes (contour)"""
		return Features._nearest_points(self._contour(cell_id1, time_id), self._contour(cell_id2, time_id))[-1] * self.scale_length

	def pair_majmaj_angle(self, time_id: int, cell_id1: int, cell_id2: int) -> float:
		"""angle between two cells major axes, between [0, pi/2]"""
		el1, el2 = self._ellipse(cell_id1, time_id), self._ellipse(cell_id2, time_id)
		maj1, maj2 = np.array([np.cos(el1.angle), np.sin(el1.angle)]), np.array([np.cos(el2.angle), np.sin(el2.angle)])
		# return np.abs(np.dot(maj1, maj2))
		return np.arccos(np.abs(np.dot(maj1, maj2)))

	def pair_majbudpt_angle(self, time_id: int, bud_id: int, candidate_id: int) -> float:
		"""angle between parent_cm->budding_point with parent major axis, between [0, pi/2]"""
		budpt = self.pair_budpt(time_id, bud_id, candidate_id)
		budpt /= np.linalg.norm(budpt)  # normalize
		el_candidate = self._ellipse(candidate_id, time_id)
		maj_candidate = np.array([ np.cos(el_candidate.angle), np.sin(el_candidate.angle) ])
		# return np.abs(np.dot(budpt, maj_candidate))
		return np.arccos(np.abs(np.dot(budpt, maj_candidate)))

	def pair_cmtocm_budmaj_angle(self, time_id: int, bud_id: int, candidate_id: int) -> float:
		"""angle between bud major axis and the cmtocm vector, between [0, pi/2]"""

		# mean angle ?
		# majs_bud, _ = self._fn_times(
		# 	lambda t: np.array([ np.cos(self._ellipse(bud_id, t).angle), np.sin(self._ellipse(bud_id, t).angle) ]),
		# 	time_id, self.budding_time
		# )
		# if len(majs_bud) < 1: return np.nan
		# maj_bud = np.mean(majs_bud, axis=0)
		# maj_bud /= np.linalg.norm(maj_bud)

		# angle @ max ecc ?
		# angle = self._ellipse_max_ecc(bud_id, time_id)[0].angle
		# maj_bud = np.array([ np.cos(angle), np.sin(angle) ])

		# angle at bud time ?
		angle = self._ellipse(bud_id, time_id).angle
		maj_bud = np.array([ np.cos(angle), np.sin(angle) ])

		cmtocm = self.pair_cmtocm(time_id, bud_id, candidate_id)
		cmtocm /= np.linalg.norm(cmtocm)
		return np.arccos(np.abs(np.dot(maj_bud, cmtocm)))

	def pair_budpt(self, time_id: int, bud_id: int, candidate_id: int) -> np.ndarray:
		"""estimated budding point of the bud relative to the parent's center of mass"""
		# WARNING : CM is in (y, x) coordinates, and budding_point in (x, y) coordinates
		return (self._budding_point(bud_id, candidate_id, time_id) - self._cm(candidate_id, time_id)[[1, 0]]) * self.scale_length

	def budcm_budpt(self, time_id: int, bud_id: int, candidate_id: int) -> np.ndarray:
		"""estimated budding point of the bud relative to the parent's center of mass"""
		# WARNING : CM is in (y, x) coordinates, and budding_point in (x, y) coordinates
		return (self._budding_point(bud_id, candidate_id, time_id) - self._cm(bud_id, time_id)[[1, 0]]) * self.scale_length

	def pair_expspeed(self, time_id: int, bud_id: int, candidate_id: int) -> float:
		"""expansion speed of the bud with respect to the parent, estimated on a few frames after given time"""
		with warnings.catch_warnings():
			warnings.filterwarnings('ignore', category=Features.FnTimesWarning)
			expdists, ok_time_ids = self._fn_times(
				lambda t: self._expansion_distance(bud_id, candidate_id, t),
				time_id, self.budding_time
			)

		if len(ok_time_ids) < 2:
			# the cell was probably flushed away
			# not enough datapoints to calculate gradient
			return np.nan

		# we need ok_time_ids, in order to account for missing frames
		expspeed = np.nanmean(np.gradient(expdists, ok_time_ids))
		return expspeed * self.scale_length / self.scale_time

	@staticmethod
	def _nearest_points(contour1: Contour, contour2: Contour) -> Tuple[np.ndarray, np.ndarray, float]:
		"""Find the pair of points between two cell contours which minimizes distance

		Parameters
		----------
		contour1 : Contour
			contour of the first cell
		contour2 : Contour
			contour of the second cell

		Returns
		-------
		point1 : numpy.ndarray (shape=(2,), dtype=int)
		point2 : numpy.ndarray (shape=(2,), dtype=int)
		dist : float
			euclidian distance between point1 and point2
		"""

		dist = scipy.spatial.distance.cdist(contour1.data, contour2.data, 'euclidean')
		i, j = np.unravel_index(dist.argmin(), dist.shape)
		nearest_point1 = contour1[i]
		nearest_point2 = contour2[j]
		min_dist = dist.min()
		return nearest_point1, nearest_point2, min_dist

	@staticmethod
	def _farthest_point_on_contour(point: np.ndarray, contour: Contour) -> np.ndarray:
		"""Return the farthest point from ``point`` on a given ``contour``

		Parameters
		----------
		point : np.ndarray, shape=(2,)
		contour : Contour

		Returns
		-------
		point : np.ndarray, shape=(2,)
			the farthest point (x, y)
		"""

		dist = scipy.spatial.distance.cdist(point[None, :], contour.data, 'euclidean')
		i, j = np.unravel_index(dist.argmax(), dist.shape)
		return contour[j]

	def _nearest_neighbours_of(self, time_id, cell_id: int, candidate_ids: Optional[List[int]] = None) -> List[int]:
		"""Returns the neighbouring cells close to ``cell_id``

		Parameters
		----------
		time_id : int
			frame index in the movie.
		cell_id : int
			cell to find the neighbours of.
		candidate_ids : Optional[List[int]], optional
			Search for neighbours only in this list, by default None.
			If ``None``, all cells in the frame (except ``cell_id``) are considered.

		Returns
		-------
		List[int]
			list of neighbouring cell ids, sorted in order of increasing distance
		"""

		if candidate_ids is None:
			candidate_ids = self.segmentation.cell_ids(time_id)

		candidate_ids = np.setdiff1d(candidate_ids, [cell_id])
		contour_target = self._contour(cell_id, time_id)
		dists = np.zeros_like(candidate_ids, dtype=float)

		for i, other_id in enumerate(candidate_ids):
			contour_other = self._contour(other_id, time_id)
			dists[i] = Features._nearest_points(contour_target, contour_other)[-1]

		# sort in the order of increasing distances
		sorted_idx = np.argsort(dists)
		candidate_ids = candidate_ids[sorted_idx]
		dists = dists[sorted_idx]

		return candidate_ids[dists <= self.nn_threshold].tolist()

	def _budding_point(self, bud_id: int, parent_id: int, time_id: int) -> np.ndarray:
		"""Compute the budding point of a bud with respect to its candidate parent

		Method :
		1. find all the points on the parent's contour which are close enough to the bud (distance less than ``self._bud_distance_max``)
		2. if there are more than 1 points, compute the average of these points ; else select the closest one

		Parameters
		----------
		bud_id : int
			id of the bud in the segmentation
		parent_id : int
			id of the candidate parent in the segmentation
		time_id : int
			frame index in the movie

		Returns
		-------
		budding_point : np.ndarray, shape=(2,)
			the guessed point (x, y) from which the bud grows
		"""

		contour_parent = self._contour(parent_id, time_id)
		contour_bud = self._contour(bud_id, time_id)
		dists = scipy.spatial.distance.cdist(contour_parent.data, contour_bud.data, 'euclidean')
		dists_mask_close = dists < self.bud_distance_max

		if np.any(dists_mask_close):
			# at least one point on the parent-bud interface
			ijs = np.argwhere(dists <= self.bud_distance_max)
			budding_point = np.mean(contour_parent[ijs[:, 0]], axis=0)
		else:
			# no point on the parent-bud interface, resort to finding smallest distance
			i, j = np.unravel_index(dists.argmin(), dists.shape)
			budding_point = contour_parent[i]
			# warnings.warn(f'no distance below {self.bud_distance_max}, found mininimum {dists.min()}')
		
		return budding_point

	class FnTimesWarning(BreadWarning):
		def __init__(self, fn: Callable, time_id: int, e: Exception):
			super().__init__(f'Could not determine call to {fn} at time_id={time_id}, got exception {e}')

	def _fn_times(self, fn: Callable[[int], R], time_id: int, num_frames: int) -> Tuple[List[R], List[int]]:
		"""Computes multiples calls to a function which takes ``time_id`` and returns the results of the successful calls

		Parameters
		----------
		fn : Callable[[int], R]
			Function to be called at each frame between ``time_id`` (inclusive) and ``time_id + num_frames`` (exclusive)
		time_id : int
			start frame
		num_frames : int
			total number of requested frames

		Returns
		-------
		results, ok_time_ids : Tuple[List[R], List[int]]
			list of results from ``fn`` calls, list of ``time_id`` for which the call succeeded
		"""

		results: List[R] = []
		ok_time_ids: List[int] = []
		frame_range = self.segmentation.request_frame_range(time_id, time_id + num_frames)
		
		for time_id_ in frame_range:
			try:
				result = fn(time_id_)
			except BreadException as e:
				warnings.warn(Features.FnTimesWarning(fn, time_id, e))
			else:
				results.append(result)
				ok_time_ids.append(time_id_)

		return results, ok_time_ids

	def _expansion_distance(self, bud_id: int, parent_id: int, time_id: int) -> float:
		"""Computes the "expansion distance" of a bud relative to a candidate cell.

		Parameters
		----------
		bud_id : int
			id of the bud in the segmentation
		parent_id : int
			id of the candidate parent in the segmentation
		time_id : int
			frame index in the movie

		Returns
		-------
		dist: float
		"""

		# bud_contour = self._contour(bud_id, time_id)
		# budding_point = self._budding_point(bud_id, parent_id, time_id)
		# point_bud_farthest = self._farthest_point_on_contour(budding_point, bud_contour)

		# return scipy.spatial.distance.cdist(budding_point[None, :], point_bud_farthest[None, :])[0, 0]

		return np.linalg.norm(self._expansion_vector(bud_id, parent_id, time_id))

	def _expansion_vector(self, bud_id: int, parent_id: int, time_id: int) -> np.ndarray:
		"""Computes the "expansion vector" (vector from budpt-to-farthest) of a bud relative to a candidate cell.

		Parameters
		----------
		bud_id : int
			id of the bud in the segmentation
		parent_id : int
			id of the candidate parent in the segmentation
		time_id : int
			frame index in the movie

		Returns
		-------
		np.ndarray (shape=2)
		"""

		bud_contour = self._contour(bud_id, time_id)
		budding_point = self._budding_point(bud_id, parent_id, time_id)
		point_bud_farthest = self._farthest_point_on_contour(budding_point, bud_contour)
		
		return point_bud_farthest - budding_point

	def _contour(self, cell_id: int, time_id: int) -> Contour:
		"""Return (possibly cached) contour for a cell"""

		# we can't use self._cached_contours.get(...) because we need to lazily compute it as needed
		
		key = (cell_id, time_id)
		if key not in self._cached_contours:
			self._cached_contours[key] = Contour.from_segmentation(self.segmentation, cell_id, time_id)
		
		return self._cached_contours[key]

	def _cm(self, cell_id: int, time_id: int) -> np.ndarray:
		"""Return (possibly cached) center of mass for a cell"""

		key = (cell_id, time_id)
		if key not in self._cached_cms:
			self._cached_cms[key] = self.segmentation.cms(time_id, [cell_id])[0]
		
		return self._cached_cms[key]

	def _ellipse(self, cell_id: int, time_id: int) -> Ellipse:
		"""Return (possibly cached) elliptical fit for a cell"""

		key = (cell_id, time_id)
		if key not in self._cached_ellipses:
			self._cached_ellipses[key] = Ellipse.from_contour(self._contour(cell_id, time_id))
		
		return self._cached_ellipses[key]

	def _ellipse_max_ecc(self, cell_id: int, time_id: int) -> Tuple[Ellipse, int]:
		"""Return the (possibly cached) ellipse of a cell which has maximal eccentricity between ``time+_id`` and ``time_id + self.budding_time`` frames
		
		Returns
		-------
		ellipse, time_id: Ellipse, int
			ellipse of maximal eccentricity, ``time_id`` of the elliptical fit
		"""

		key = (cell_id, time_id, self.budding_time)
		if key not in self._cached_ellipses_max_ecc:
			# els should never be empty, since there should always be an elliptical fit
			# on the bud in the first frame
			els, ok_time_ids = self._fn_times(
				lambda t: self._ellipse(cell_id, t),
				time_id, self.budding_time
			)
			eccs = [ el.ecc for el in els ]
			imax = eccs.index(max(eccs))
			el = els[imax]
			time_id_ = ok_time_ids[imax]
			self._cached_ellipses_max_ecc[key] = (el, time_id_)

		return self._cached_ellipses_max_ecc[key]