from enum import IntEnum
import numpy as np
import scipy.ndimage, scipy.spatial
import warnings
import cv2 as cv
from pathlib import Path
from typing import Tuple, Union, Optional, List
from dataclasses import dataclass
from ._utils import load_npz
from ._exception import BreadException, BreadWarning

__all__ = ['Lineage', 'Microscopy', 'Segmentation', 'Contour', 'Ellipse']

@dataclass
class Lineage:
	"""Store lineage relations for cells in a movie."""

	parent_ids: np.ndarray
	bud_ids: np.ndarray
	time_ids: np.ndarray

	class SpecialParentIDs(IntEnum):
		"""Special parent IDs attributed in lineages to specify exceptions.
		
		Attributes
		----------
		PARENT_OF_ROOT : int = -1
			parent of a cell that already exists in first frame of colony
		PARENT_OF_EXTERNAL : int = -2
			parent of a cell that does not belong to the colony
		NO_GUESS : int = -3
			parent of cell for which the algorithm failed to guess
		"""

		PARENT_OF_ROOT: int = -1
		"""parent of a cell that already exists in first frame of colony"""
		PARENT_OF_EXTERNAL: int = -2
		"""parent of a cell that does not belong to the colony"""
		NO_GUESS: int = -3
		"""parent of cell for which the algorithm failed to guess"""

		@staticmethod
		def is_special(cell_id):
			return cell_id in [sp_id.value for sp_id in Lineage.SpecialParentIDs]

	def __post_init__(self):
		assert self.parent_ids.ndim == 1, '`parent_ids` should have 1 dimension'
		assert self.bud_ids.ndim == 1, '`bud_ids` should have 1 dimension'
		assert self.time_ids.ndim == 1, '`time_ids` should have 1 dimension'
		assert self.parent_ids.shape == self.bud_ids.shape == self.time_ids.shape, '`parent_ids`, `bud_ids`, `time_ids` should have the same shape'

		if not np.issubdtype(self.time_ids.dtype, np.integer):
			warnings.warn(f'Lineage.time_ids initialized with non-int, {self.time_ids.dtype} used.')
		if not np.issubdtype(self.bud_ids.dtype, np.integer):
			warnings.warn(f'Lineage.bud_ids initialized with non-int, {self.bud_ids.dtype} used.')
		if not np.issubdtype(self.parent_ids.dtype, np.integer):
			warnings.warn(f'Lineage.parent_ids initialized with non-int, {self.parent_ids.dtype} used.')

	def __len__(self):
		return len(self.time_ids)

	def extended_budding_events(self, duration: int, time_id_max: Optional[int] = None, return_dts: Optional[bool] = False) -> Tuple['Lineage', List[int]]:
		"""Returns a modified copy of itself, where each budding event now spans ``duration`` frames, up to a maximum of ``time_id_max``

		Parameters
		----------
		duration : int
			Duration in frames of each budding event in the lineage
		time_id_max : Optional[int], optional
			Maximum ``time_id`` allowed, by default None.
			Recommended setting this to ``len(segmentation)``.
		return_dts : bool, optional
			Return number of frames since the budding as the second element to the return tuple

		Returns
		-------
		extended_lineage: Lineage
		dts: List[int], only when ``return_dts == True``
			number of frames since the budding event
		"""

		parent_ids, bud_ids, time_ids = [], [], []
		dts: List[int] = []

		for parent_id, bud_id, time_id in self:
			for dt in range(0, duration):
				if time_id_max is not None and time_id + dt > time_id_max: continue
				parent_ids.append(parent_id)
				bud_ids.append(bud_id)
				time_ids.append(time_id+dt)
				dts.append(dt)

		lin = Lineage(np.array(parent_ids, dtype=int), np.array(bud_ids, dtype=int), np.array(time_ids, dtype=int))

		if return_dts:
			return lin, dts
		return lin

	def only_budding_events(self) -> 'Lineage':
		"""Returns a copy of itself, where only budding event (i.e. non special parent ids) are present

		Returns
		-------
		Lineage
			new lineage with only budding events
		"""

		mask = np.array([ not Lineage.SpecialParentIDs.is_special(parent_id) for parent_id in self.parent_ids ])

		return Lineage(
			parent_ids=self.parent_ids[mask].copy(),
			bud_ids=self.bud_ids[mask].copy(),
			time_ids=self.time_ids[mask].copy(),
		)

	def __iter__(self) -> 'Lineage':
		self._idx_iter = 0
		return self

	def __next__(self) -> Tuple[int, int, int]:
		self._idx_iter += 1

		if self._idx_iter-1 == len(self):
			raise StopIteration()

		return self.parent_ids[self._idx_iter-1], self.bud_ids[self._idx_iter-1], self.time_ids[self._idx_iter-1]

	def save_csv(self, filepath: Path):
		np.savetxt(
			filepath,
			np.array((self.parent_ids, self.bud_ids, self.time_ids), dtype=int).T,
			delimiter=',', header='parent_id,bud_id,time_index',
			fmt='%.0f'  # floating point to support nan values
		)

	@staticmethod
	def from_csv(filepath: Path) -> 'Lineage':
		parent_ids, bud_ids, time_ids = np.genfromtxt(filepath, skip_header=True, delimiter=',', unpack=True, dtype=int)
		if not isinstance(parent_ids, np.ndarray):  # in files with one line, genfromtxt returns a float, not a numpy array
			parent_ids = np.array((parent_ids,), dtype=int)
		if not isinstance(bud_ids, np.ndarray):  # in files with one line, genfromtxt returns a float, not a numpy array
			bud_ids = np.array((bud_ids,), dtype=int)
		if not isinstance(time_ids, np.ndarray):  # in files with one line, genfromtxt returns a float, not a numpy array
			time_ids = np.array((time_ids,), dtype=int)
		return Lineage(
			parent_ids=parent_ids,
			bud_ids=bud_ids,
			time_ids=time_ids
		)

@dataclass
class Microscopy:
	"""Store a raw microscopy movie.
	
	data : numpy.ndarray (shape=(T, W, H))
		T : number of timeframes
		W, H : shape of the images
	"""

	data: np.ndarray

	def __getitem__(self, index):
		return self.data[index]
		
	def __len__(self):
		return len(self.data)

	def __post_init__(self):
		if self.data.ndim == 2:
			warnings.warn('Microscopy was given data with 2 dimensions, adding an empty dimension for time.')
			self.data = self.data[None, ...]

		assert self.data.ndim == 3, 'Microscopy data should have 3 dimensions, expected shape (time, height, width)'

	def __repr__(self) -> str:
		return 'Microscopy(num_frames={}, frame_height={}, frame_width={})'.format(*self.data.shape)

	@staticmethod
	def from_tiff(filepath: Path) -> 'Microscopy':
		import tifffile
		data = tifffile.imread(filepath)
		return Microscopy(data=data)

	@staticmethod
	def from_npzs(filepaths: Union[Path, List[Path]]) -> 'Microscopy':
		"""Loads a microscopy movie from a list `.npz` files. Each `.npz` file stores one 2D array, corresponding to a frame.

		Parameters
		----------
		filepaths : Union[Path, list[Path]]
			Paths to the `.npz` files. If only a `Path` is given, assumes one frame in movie.

		Returns
		-------
		Microscopy
		"""

		if not isinstance(filepaths, list):
			filepaths = [filepaths]
		data = np.array(load_npz(filepaths))
		return Microscopy(data=data)


@dataclass
class Segmentation:
	"""Store a segmentation movie.

	Each image stores ids corresponding to the mask of the corresponding cell.

	data : numpy.ndarray (shape=(T, W, H))
		T : number of timeframes
		W, H : shape of the images
	"""

	data: np.ndarray

	def __getitem__(self, index):
		return self.data[index]

	def __len__(self):
		return len(self.data)

	def __post_init__(self):
		if self.data.ndim == 2:
			warnings.warn('Microscopy was given data with 2 dimensions, adding an empty dimension for time.')
			self.data = self.data[None, ...]

		assert self.data.ndim == 3

	def __repr__(self) -> str:
		return 'Segmentation(num_frames={}, frame_height={}, frame_width={})'.format(*self.data.shape)

	def cell_ids(self, time_id: Optional[int] = None, background_id: Optional[int]=0) -> List[int]:
		"""Returns cell ids from a segmentation

		Parameters
		----------
		time_id : int or None, optional
			frame index in the movie. If None, returns all the cellids encountered in the movie
		background_id : int or None, optional
			if not None, remove id `background_id` from the cell ids

		Returns
		-------
		List[int]
			cell ids contained in the segmentation, in sorted order
		"""

		if time_id is None:
			all_ids = np.unique(self.data.flat)
		else:
			all_ids = np.unique(self.data[time_id].flat)

		if background_id is not None:
			return list(all_ids[all_ids != background_id])
		else:
			return list(all_ids)

	def cms(self, time_id: int, cell_ids: Optional[List[int]] = None) -> np.ndarray:
		"""Returns centers of mass of cells in a segmentation

		Parameters
		----------
		time_id : int
			Frame index in the movie
		cell_ids : List[int]
			List of cell ids for which to compute the centers of mass, by default None.
			If ``None``, ``cell_ids`` becomes all the cells in the frame

		Returns
		-------
		array-like of shape (ncells, 2)
			(y, x) coordinates of the centers of mass of each cell
		"""

		if cell_ids is None:
			cell_ids = self.cell_ids(time_id)
		cms = np.zeros((len(cell_ids), 2))

		for i, cell_id in enumerate(cell_ids):
			cms[i] = scipy.ndimage.center_of_mass(self.data[time_id] == cell_id)

		return cms

	def find_buds(self) -> Lineage:
		"""Return IDs of newly created cells

		Returns
		-------
		lineage: Lineage
			initialized lineage, with nan parent ids
		"""

		bud_ids, time_ids = [], []

		for idt in range(len(self)):
			cellids = self.cell_ids(idt)
			diffids = np.setdiff1d(cellids, bud_ids, assume_unique=True)

			bud_ids += list(diffids)
			time_ids += [idt] * len(diffids)

		return Lineage(
			parent_ids=np.full(len(bud_ids), Lineage.SpecialParentIDs.NO_GUESS.value, dtype=int),
			bud_ids=np.array(bud_ids, dtype=int),
			time_ids=np.array(time_ids, dtype=int)
		)

	def request_frame_range(self, time_id_from: int, time_id_to: int):
		"""Generate a range of valid frames going from time_id_from (inclusive) to time_id_to (exclusive)

		Parameters
		----------
		time_id_from : int
			index of the first frame
		time_id_to : int
			index of the last frame - 1

		Returns
		-------
		frame_range : range
		"""
		
		assert time_id_from < time_id_to, 'time_id_from should be strictly less than time_id_to'
		num_frames = time_id_to - time_id_from
		num_frames_available = min(max(0, len(self) - time_id_from), num_frames)
		return range(time_id_from, time_id_from + num_frames_available)

	def crop(self, x: int, y: int, w: int, h: int) -> 'Segmentation':
		"""Return a cropped version of the segmentation"""
		return Segmentation(self.data[:, y:y+h, x:x+w])

	@staticmethod
	def from_h5(filepath: Path, fov: str = 'FOV0', load_frames: Optional[List[int]] = None) -> 'Segmentation':
		"""Read h5 segmentation data from YeaZ"""
		import h5py
		file = h5py.File(filepath, 'r')

		if load_frames is None:
			load_frames = list(range(len(file[fov].keys())))
		
		keys = [ f'T{n}' for n in load_frames ]

		imgs = np.zeros((len(keys), *file[fov]['T0'].shape), dtype=int)
		for i, key in enumerate(keys):
			imgs[i] = np.array(file[fov][key])

		file.close()
		return Segmentation(imgs)

	def write_h5(self, filepath: Path, fov: str = 'FOV0'):
		"""Write h5 segmentation to read using YeaZ"""
		import h5py
		file = h5py.File(filepath, 'w')
		file.create_group(fov)
		for i in range(len(self)):
			file.create_dataset(f'{fov}/T{i}', data=self.data[i], compression='gzip')
		file.close()

	@staticmethod
	def from_npzs(filepaths: Union[Path, List[Path]]) -> 'Segmentation':
		"""Loads a segmentation movie from a list `.npz` files. Each `.npz` file stores one 2D array, corresponding to a frame.

		Parameters
		----------
		filepaths : Union[Path, list[Path]]
			Paths to the `.npz` files. If only a `Path` is given, assumes one frame in movie.

		Returns
		-------
		Segmentation
		"""

		if not isinstance(filepaths, list):
			filepaths = [filepaths]
		data = np.array(load_npz(filepaths))
		return Segmentation(data=data)


@dataclass
class Contour:
	"""Stores indices of the contour of the cell
	
	data : numpy.ndarray (shape=(N, 2))
		Stores a list of (x, y) points corresponding to indices of the contour.
		Warning : images are indexes as `img[y, x]`, so use `img[contour[:, 1], contour[:, 0]]`
	"""

	data: np.ndarray

	def __getitem__(self, index):
		return self.data[index]
		
	def __len__(self):
		return len(self.data)

	class InvalidContourException(BreadException):
		def __init__(self, mask: np.ndarray):
			super().__init__(f'Unable to extract a contour from the mask. Did you check visually if the mask is connected, or large enough (found {len(np.nonzero(mask))} nonzero pixels) ?')

	class CellMissingException(BreadException):
		def __init__(self, cell_id: int, time_id: int):
			super().__init__(f'Unable to find cell_id={cell_id} at time_id={time_id} in the segmentation.')

	class MultipleContoursWarning(BreadWarning):
		def __init__(self, num: int) -> None:
			super().__init__(f'OpenCV returned multiple contours, {num} found.')

	def __post_init__(self):
		assert self.data.ndim == 2, 'Contour expected data with 2 dimensions, with shape (N, 2)'
		assert self.data.shape[1] == 2, 'Contour expected data with shape (N, 2)'
		
		if not np.issubdtype(self.data.dtype, np.integer):
			warnings.warn(f'Contour initialized with non-integer, {self.data.dtype} used.')

	def __repr__(self) -> str:
		return 'Contour(num_points={})'.format(self.data.shape[0])

	@property
	def area(self) -> float:
		"""Compute the area of the contour

		Returns
		-------
		float
			Area of the contour, in px²
		"""

		return cv.contourArea(self.data)

	@staticmethod
	def from_segmentation(seg: Segmentation, cell_id: int, time_id: int) -> 'Contour':
		"""Return the contour of a cell at a frame in the segmentation

		Parameters
		----------
		seg : Segmentation
		cell_id : int
		time_id : int

		Returns
		-------
		Contour
			
		Raises
		------
		Contour.InvalidContourException
			Raised if the cell mask is invalid (often too small or disjointed)
		Contour.MissingCellException
			``cell_id`` was not found in the segmentation at ``time_id``
		"""

		mask = seg[time_id] == cell_id

		if not mask.any():
			raise Contour.CellMissingException(cell_id, time_id)

		# TODO : check connectivity
		contours_cv, *_ = cv.findContours(mask.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

		if len(contours_cv) == 0:
			raise Contour.InvalidContourException(mask)
		
		if len(contours_cv) > 1:
			warnings.warn(Contour.MultipleContoursWarning(len(contours_cv)))

		contour = max(contours_cv, key=cv.contourArea)  # return the contour with the largest area
		return Contour(
			np.vstack(contour).squeeze()  # convert to numpy array with correct shape and remove unneeded dimensions
		)


@dataclass
class Ellipse:
	"""Store properties of an ellipse."""

	x: float
	y: float
	r_maj: float
	r_min: float
	angle: float

	@staticmethod
	def from_contour(contour: Contour):
		xy, wh, angle_min = cv.fitEllipse(contour.data)
		r_min, r_maj = wh[0]/2, wh[1]/2
		assert r_min <= r_maj
		angle_maj = np.mod(np.deg2rad(angle_min) + np.pi/2, np.pi)  # angle of the major axis
		return Ellipse(xy[0], xy[1], r_maj, r_min, angle_maj)

	@property
	def r_equiv(self) -> float:
		"""radius of the circle with same area as ellipse : r=sqrt(a*b)"""
		return np.sqrt(self.r_maj*self.r_min)

	@property
	def ecc(self) -> float:
		"""eccentricity"""
		return np.sqrt(1 - (self.r_min/self.r_maj)**2)