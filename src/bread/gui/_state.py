from PySide6.QtCore import QObject, Signal, Slot
from bread.data import Lineage, Microscopy, Segmentation, SegmentationFile
from typing import Optional, List
from dataclasses import dataclass, field
from pathlib import Path
from _utils import clamp

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('bread.state')

__all__ = ['AppState', 'APP_STATE']

class AppState(QObject):
	update_segmentation_data = Signal(Segmentation)
	update_microscopy_data = Signal(Microscopy)
	update_budneck_data = Signal(Microscopy)
	update_nucleus_data = Signal(Microscopy)
	update_current_lineage_data = Signal(Lineage)
	update_add_lineage_data = Signal(Lineage, Path)
	update_fov_list = Signal(list)
	update_fov = Signal(str)
	update_segmentation_opacity = Signal(float)
	update_microscopy_opacity = Signal(float)
	update_budneck_opacity = Signal(float)
	update_nucleus_opacity = Signal(float)
	update_show_ids = Signal(bool)
	update_show_lineage_graph = Signal(bool)
	update_clicked_cellid = Signal(int)
	update_frame_index = Signal(int)
	update_frames_max = Signal(int)
	update_centered_cellid = Signal(str, int, int)
	closing = Signal()

	@dataclass
	class AppData:
		parent: 'AppState' = field(repr=False)
		segmentation: Optional[SegmentationFile] = None
		microscopy: Optional[Microscopy] = None
		budneck: Optional[Microscopy] = None
		nucleus: Optional[Microscopy] = None
		current_lineage: Optional[Lineage] = None
		# lineages: List[Lineage] = field(default_factory=list)

		@property
		def frames_max(self) -> int:
			try:
				max_micro = APP_STATE.data.microscopy.get_frame_count(fov=APP_STATE.values.fov) if APP_STATE.data.microscopy else None
				max_seg = APP_STATE.data.segmentation[APP_STATE.values.fov].get_frame_count() if APP_STATE.data.segmentation else None
				max_budneck = APP_STATE.data.budneck.get_frame_count(fov=APP_STATE.values.fov) if APP_STATE.data.budneck else None
				max_nucleus = APP_STATE.data.nucleus.get_frame_count(fov=APP_STATE.values.fov) if APP_STATE.data.nucleus else None
				min_value = min(x for x in [max_micro, max_budneck, max_seg, max_nucleus] if x is not None)
				return min_value
			except Exception as e:
				logger.exception("An exception occurred: %s", str(e))
				return 0
		
		@property
		def fov_list(self) -> List[str]:
			try:
				microscopy_fov = self.microscopy.field_of_views if self.microscopy else ["FOV0"]
				segmentation_fov = self.segmentation.field_of_views if self.segmentation else ["FOV0"]
				budneck_fov = self.budneck.field_of_views if self.budneck else ["FOV0"]
				nucleus_fov = self.nucleus.field_of_views if self.nucleus else ["FOV0"]
				fov_list = list(set(microscopy_fov + segmentation_fov + budneck_fov + nucleus_fov))
				fov_list.sort()
				return fov_list
			except Exception as e:
				logger.exception("An exception occurred: %s", str(e))
				return []

		@property
		def frames_homogeneous(self) -> bool:
			return len(self.segmentation[fov] or []) == len(self.microscopy[fov] or []) == len(self.nucleus[fov] or [])

		def valid_frameid(self, frameid) -> bool:
			return 0 <= frameid < self.frames_max

		def valid_cellid(self, fov:str, cellid: int, timeid: int) -> bool:
			return self.segmentation[fov] is not None and cellid in self.segmentation[fov].cell_ids(timeid)
		
		def get_coresponding_channel(self, type):
			if type == "Microscopy":
				return self.microscopy.get_channel() if self.microscopy else "None"
			elif type == "Budneck":
				return self.budneck.get_channel() if self.budneck else "None"
			elif type == "Nucleus":
				return self.nucleus.get_channel() if self.nucleus else "None"
			elif type == "Segmentation":
				return ""
		
		def get_coresponding_filepath(self, type):
			try:
				if type == "Microscopy":
					return self.microscopy.get_file_path().split('/')[-1] if self.microscopy else "No file loaded"
				elif type == "Budneck":
					return self.budneck.get_file_path().split('/')[-1] if self.budneck else "No file loaded"
				elif type == "Nucleus":
					return self.nucleus.get_file_path().split('/')[-1] if self.nucleus else "No file loaded"
				elif type == "Segmentation":
					return self.segmentation.get_file_path().split('/')[-1] if self.segmentation else "No file loaded"
			except Exception as e:
				return " "

	@dataclass
	class AppValues:
		parent: 'AppState' = field(repr=False)
		show_ids: bool = False
		show_lineage_graph: bool = False
		clicked_cellid: int = 0
		frame_index: int = 0
		fov: str = "FOV0"
		channel: str = "Channel0"
		opacity_segmentation: float = 1
		opacity_microscopy: float = 1
		opacity_budneck: float = 1
		opacity_nucleus: float = 1

	def __init__(self, parent: Optional[QObject] = None) -> None:
		super().__init__(parent)

		self.data = AppState.AppData(self)
		self.values = AppState.AppValues(self)

	def __repr__(self) -> str:
		return f'AppState({self.data}, {self.values})'

	@Slot(bool)
	def set_show_ids(self, v: bool) -> None:
		self.values.show_ids = v
		self.update_show_ids.emit(self.values.show_ids)

	@Slot(bool)
	def set_show_lineage_graph(self, v: bool) -> None: 
		self.values.show_lineage_graph = v
		self.update_show_lineage_graph.emit(self.values.show_lineage_graph)

	@Slot(int)
	def set_clicked_cellid(self, cellid: int) -> None:
		self.values.clicked_cellid = cellid
		self.update_clicked_cellid.emit(self.values.clicked_cellid)

	@Slot(int)
	def set_frame_index(self, index: int) -> None:
		self.values.frame_index = clamp(index, 0, self.data.frames_max-1)
		self.update_frame_index.emit(self.values.frame_index)

	@Slot(str)
	def set_fov(self, fov: str) -> None:
		self.values.fov = APP_STATE.data.fov_list[fov]
		self.update_fov.emit(self.data.fov_list)

	@Slot(float)
	def set_opacity_segmentation(self, opacity: float) -> None:
		self.values.opacity_segmentation = clamp(opacity, 0, 1)
		self.update_segmentation_opacity.emit(self.values.opacity_segmentation)

	@Slot(float)
	def set_opacity_microscopy(self, opacity: float) -> None:
		self.values.opacity_microscopy = clamp(opacity, 0, 1)
		self.update_microscopy_opacity.emit(self.values.opacity_microscopy)

	@Slot(float)
	def set_opacity_budneck(self, opacity: float) -> None:
		self.values.opacity_budneck = clamp(opacity, 0, 1)
		self.update_budneck_opacity.emit(self.values.opacity_budneck)

	@Slot(float)
	def set_opacity_nucleus(self, opacity: float) -> None:
		self.values.opacity_nucleus = clamp(opacity, 0, 1)
		self.update_nucleus_opacity.emit(self.values.opacity_nucleus)

	@Slot(Microscopy)
	def set_microscopy_data_from_nd2(self, microscopy: Optional[Microscopy], channel_mapper:List) -> None:
		for type in channel_mapper:
			if type == 'Brightfield/Phase Contrast':
				self.set_microscopy_data(microscopy.get_microscopy_per_channel(channel_mapper[type]))
			elif type == 'Budneck':
				self.set_budneck_data(microscopy.get_microscopy_per_channel(channel_mapper[type]))
			elif type == 'Nucleus':
				self.set_nucleus_data(microscopy.get_microscopy_per_channel(channel_mapper[type]))
			elif type == 'Other':
				pass
		

	@Slot(Microscopy)
	def set_microscopy_data(self, microscopy: Optional[Microscopy]) -> None:
		self.data.microscopy = microscopy
		self.values.frame_index = 0
		self.values.fov = microscopy.field_of_views[0] if microscopy.field_of_views else "FOV0"
		self.update_microscopy_data.emit(self.data.microscopy.get_frames(self.values.fov))
		self.update_frames_max.emit(self.data.frames_max)
		self.update_fov_list.emit(self.data.fov_list)
		self.update_frame_index.emit(0)

	@Slot(Microscopy)
	def set_budneck_data(self, budneck: Optional[Microscopy]) -> None:
		self.data.budneck = budneck
		self.values.frame_index = 0
		self.values.fov = budneck.field_of_views[0] if budneck.field_of_views else "FOV0"
		self.update_budneck_data.emit(self.data.budneck.get_frames(self.values.fov))
		self.update_frames_max.emit(self.data.frames_max)
		self.update_fov_list.emit(self.data.fov_list)
		self.update_frame_index.emit(0)

	@Slot(Microscopy)
	def set_nucleus_data(self, nucleus: Optional[Microscopy]) -> None:
		self.data.nucleus = nucleus
		self.values.frame_index = 0
		self.values.fov = nucleus.field_of_views[0] if nucleus.field_of_views else "FOV0"
		self.update_nucleus_data.emit(self.data.nucleus.get_frames(self.values.fov))
		self.update_frames_max.emit(self.data.frames_max)
		self.update_fov_list.emit(self.data.fov_list)
		self.update_frame_index.emit(0)

	@Slot(Segmentation)
	def set_segmentation_data(self, segmentation: Optional[SegmentationFile]) -> None:
		self.data.segmentation = segmentation
		self.values.frame_index = 0
		self.values.fov = segmentation.field_of_views[0] if segmentation.field_of_views else "FOV0"
		self.update_segmentation_data.emit(self.data.segmentation[self.values.fov].data)
		self.update_frames_max.emit(self.data.frames_max)
		self.update_fov_list.emit(self.data.fov_list)
		self.update_frame_index.emit(0)

	@Slot(Lineage)
	def set_current_lineage_data(self, lineage: Optional[Lineage]) -> None:
		self.data.current_lineage = lineage
		self.update_current_lineage_data.emit(self.data.current_lineage)

	@Slot(Lineage, Path)
	def add_lineage_data(self, lineage: Lineage, filepath: Optional[Path] = None) -> None:
		# lineage is not stored, because ownership is in the table
		# maybe table updates should directly update the lineage here ?
		self.update_add_lineage_data.emit(lineage, filepath)

	@Slot(int, int)
	def set_centered_cellid(self, timeid: int, cellid: int) -> None:
		if not self.data.valid_cellid(cellid, self.values.fov, timeid):
			return

		self.update_centered_cellid.emit(self.values.fov, timeid, cellid)

APP_STATE = AppState()