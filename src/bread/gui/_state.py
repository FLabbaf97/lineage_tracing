from PySide6.QtCore import QObject, Signal, Slot
from bread.data import Lineage, Microscopy, Segmentation
from typing import Optional, List
from dataclasses import dataclass, field
from pathlib import Path
from ._utils import clamp

__all__ = ['AppState', 'APP_STATE']

class AppState(QObject):
	update_segmentation_data = Signal(Segmentation)
	update_microscopy_data = Signal(Microscopy)
	update_budneck_data = Signal(Microscopy)
	update_nucleus_data = Signal(Microscopy)
	update_current_lineage_data = Signal(Lineage)
	update_add_lineage_data = Signal(Lineage, Path)
	update_segmentation_opacity = Signal(float)
	update_microscopy_opacity = Signal(float)
	update_budneck_opacity = Signal(float)
	update_nucleus_opacity = Signal(float)
	update_show_ids = Signal(bool)
	update_show_lineage_graph = Signal(bool)
	update_clicked_cellid = Signal(int)
	update_frame_index = Signal(int)
	update_frames_max = Signal(int)
	update_centered_cellid = Signal(int, int)
	closing = Signal()

	@dataclass
	class AppData:
		parent: 'AppState' = field(repr=False)
		segmentation: Optional[Segmentation] = None
		microscopy: Optional[Microscopy] = None
		budneck: Optional[Microscopy] = None
		nucleus: Optional[Microscopy] = None
		current_lineage: Optional[Lineage] = None
		# lineages: List[Lineage] = field(default_factory=list)

		@property
		def frames_max(self) -> int:
			try:
				m = min(map(lambda x: len(x), filter(lambda x: x is not None, [self.segmentation, self.microscopy, self.budneck, self.nucleus])))
			except ValueError:
				# min() arg is an empty sequence
				m = 0
			return m

		@property
		def frames_homogeneous(self) -> bool:
			return len(self.segmentation or []) == len(self.microscopy or []) == len(self.nucleus or [])

		def valid_frameid(self, frameid) -> bool:
			return 0 <= frameid < self.frames_max

		def valid_cellid(self, cellid: int, timeid: int) -> bool:
			return self.segmentation is not None and cellid in self.segmentation.cell_ids(timeid)

	@dataclass
	class AppValues:
		parent: 'AppState' = field(repr=False)
		show_ids: bool = False
		show_lineage_graph: bool = False
		clicked_cellid: int = 0
		frame_index: int = 0
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

	@Slot(Segmentation)
	def set_segmentation_data(self, segmentation: Optional[Segmentation]) -> None:
		self.data.segmentation = segmentation
		self.update_segmentation_data.emit(self.data.segmentation)
		self.update_frames_max.emit(self.data.frames_max)

	@Slot(Microscopy)
	def set_microscopy_data(self, microscopy: Optional[Microscopy]) -> None:
		self.data.microscopy = microscopy
		self.update_microscopy_data.emit(self.data.microscopy)
		self.update_frames_max.emit(self.data.frames_max)

	@Slot(Microscopy)
	def set_budneck_data(self, budneck: Optional[Microscopy]) -> None:
		self.data.budneck = budneck
		self.update_budneck_data.emit(self.data.budneck)
		self.update_frames_max.emit(self.data.frames_max)

	@Slot(Microscopy)
	def set_nucleus_data(self, nucleus: Optional[Microscopy]) -> None:
		self.data.nucleus = nucleus
		self.update_nucleus_data.emit(self.data.nucleus)
		self.update_frames_max.emit(self.data.frames_max)

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
		if not self.data.valid_cellid(cellid, timeid):
			return

		self.update_centered_cellid.emit(timeid, cellid)

APP_STATE = AppState()