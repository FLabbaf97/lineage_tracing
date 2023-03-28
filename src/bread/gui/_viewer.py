import numpy as np
from PySide6 import QtGui, QtWidgets, QtCore
from PySide6.QtWidgets import QWidget, QMenuBar, QMainWindow, QVBoxLayout, QLabel, QHBoxLayout, QGridLayout, QPushButton, QCheckBox, QSlider, QTableWidget, QTableWidgetItem, QTabWidget, QGroupBox, QSpinBox, QFileDialog, QMessageBox
from PySide6.QtGui import QIcon
from PySide6.QtCore import QObject, Signal, Slot
from typing import Optional, List
from pathlib import Path
import pyqtgraph as pg
from bread.data import Lineage, Microscopy, Segmentation
from ._state import APP_STATE
from ._utils import lerp

__all__ = ['Viewer']

pg.setConfigOption('imageAxisOrder', 'row-major')

class Layer(QGroupBox):
	def __init__(self, parent: Optional[QWidget] = None, *args, **kwargs) -> None:
		super().__init__(parent, *args, **kwargs)

		self.openbtn = QPushButton('Open')
		self.openbtn.setIcon(QIcon(str(Path(__file__).parent / 'fugue-icons-3.5.6' / 'icons-shadowless' / 'folder-open-image.png')))
		self.opacityslider = QSlider(QtCore.Qt.Horizontal)
		self.opacityslider.setMinimum(0)
		self.opacityslider.setMaximum(10)
		self.opacityslider.setSingleStep(1)
		self.opacityslider.setValue(10)
		self.opacitysliderlabel = QLabel('Opacity')
		self.opacity = QWidget(self)
		self.opacity.setLayout(QHBoxLayout())
		self.opacity.layout().addWidget(self.opacitysliderlabel)
		self.opacity.layout().addWidget(self.opacityslider)
		self.setLayout(QVBoxLayout())
		self.layout().addWidget(self.openbtn)
		self.layout().addWidget(self.opacity)

		self.layout().setContentsMargins(0, 0, 0, 0)


class Layers(QWidget):
	# TODO : time shape mismatch warning
	# TODO : shape checking for data
	
	def __init__(self, parent: Optional[QWidget] = None) -> None:
		super().__init__(parent)

		self.segmentation = Layer(title='Segmentation')
		self.segmentation.opacityslider.valueChanged.connect(lambda val: APP_STATE.set_opacity_segmentation(lerp(val, self.segmentation.opacityslider.minimum(), self.segmentation.opacityslider.maximum(), 0, 1)))
		self.segmentation.openbtn.clicked.connect(self.file_open_segmentation)
		self.microscopy = Layer(title='Microscopy')
		self.microscopy.opacityslider.valueChanged.connect(lambda val: APP_STATE.set_opacity_microscopy(lerp(val, self.microscopy.opacityslider.minimum(), self.microscopy.opacityslider.maximum(), 0, 1)))
		self.microscopy.openbtn.clicked.connect(self.file_open_microscopy)
		self.budneck = Layer(title='Budneck')
		self.budneck.opacityslider.valueChanged.connect(lambda val: APP_STATE.set_opacity_budneck(lerp(val, self.budneck.opacityslider.minimum(), self.segmentation.opacityslider.maximum(), 0, 1)))
		self.budneck.openbtn.clicked.connect(self.file_open_budneck)
		self.nucleus = Layer(title='Nucleus')
		self.nucleus.opacityslider.valueChanged.connect(lambda val: APP_STATE.set_opacity_nucleus(lerp(val, self.nucleus.opacityslider.minimum(), self.segmentation.opacityslider.maximum(), 0, 1)))
		self.nucleus.openbtn.clicked.connect(self.file_open_nucleus)
		self.setLayout(QVBoxLayout())
		self.layout().setAlignment(QtCore.Qt.AlignTop)
		self.layout().setContentsMargins(0, 0, 0, 0)
		self.layout().addWidget(self.segmentation)
		self.layout().addWidget(self.microscopy)
		self.layout().addWidget(self.budneck)
		self.layout().addWidget(self.nucleus)
		self.setFixedWidth(200)
		self.layout().setContentsMargins(0, 0, 0, 0)

	@Slot()
	def file_open_segmentation(self):
		filepath, filefilter = QFileDialog.getOpenFileName(self, 'Open segmentation', './', 'Segmentation files (*.h5)')
		filepath = Path(filepath)
		if filepath.is_dir():
			return  # user did not select anything
		
		# TODO : proper file opening support in bread.data
		# something like Segmentation.from_filepath_autodetect
		if filepath.suffix in ['.h5', '.hd5']:
			segmentation = Segmentation.from_h5(filepath)
		else:
			raise RuntimeError(f'Unsupported extension : {filepath.suffix}')

		APP_STATE.set_segmentation_data(segmentation)

	@Slot()
	def file_open_microscopy(self):
		# yes, i am repeating myself, but function binding would be overkill and this is more explicit
		filepath, filefilter = QFileDialog.getOpenFileName(self, 'Open microscopy', './', 'Microscopy files (*.tiff, *.tif)')
		filepath = Path(filepath)
		if filepath.is_dir():
			return  # user did not select anything

		if filepath.suffix in ['.tif', '.tiff']:
			microscopy = Microscopy.from_tiff(filepath)
		else:
			raise RuntimeError(f'Unsupported extension : {filepath.suffix}')

		APP_STATE.set_microscopy_data(microscopy)

	@Slot()
	def file_open_budneck(self):
		# yes, i am repeating myself, but function binding would be overkill and this is more explicit
		filepath, filefilter = QFileDialog.getOpenFileName(self, 'Open budneck channel', './', 'Budneck microscopy files (*.tiff, *.tif)')
		filepath = Path(filepath)
		if filepath.is_dir():
			return  # user did not select anything

		if filepath.suffix in ['.tif', '.tiff']:
			budneck = Microscopy.from_tiff(filepath)
		else:
			raise RuntimeError(f'Unsupported extension : {filepath.suffix}')

		APP_STATE.set_budneck_data(budneck)

	@Slot()
	def file_open_nucleus(self):
		# yes, i am repeating myself, but function binding would be overkill and this is more explicit
		filepath, filefilter = QFileDialog.getOpenFileName(self, 'Open nucleus channel', './', 'Nucleus microscopy files (*.tiff, *.tif)')
		filepath = Path(filepath)
		if filepath.is_dir():
			return  # user did not select anything

		if filepath.suffix in ['.tif', '.tiff']:
			nucleus = Microscopy.from_tiff(filepath)
		else:
			raise RuntimeError(f'Unsupported extension : {filepath.suffix}')

		APP_STATE.set_nucleus_data(nucleus)


class Timeline(QWidget):
	def __init__(self, parent: Optional[QWidget] = None) -> None:
		super().__init__(parent)

		self.prevbtn = QPushButton('Previous frame')
		self.prevbtn.setIcon(QIcon(str(Path(__file__).parent / 'fugue-icons-3.5.6' / 'icons-shadowless' / 'arrow-180.png')))
		self.prevbtn.clicked.connect(lambda: APP_STATE.set_frame_index(APP_STATE.values.frame_index-1))
		self.nextbtn = QPushButton('Next frame')
		self.nextbtn.setIcon(QIcon(str(Path(__file__).parent / 'fugue-icons-3.5.6' / 'icons-shadowless' / 'arrow.png')))
		self.nextbtn.clicked.connect(lambda: APP_STATE.set_frame_index(APP_STATE.values.frame_index+1))
		
		self.framespinbox = QSpinBox()
		self.framespinbox.setMinimum(0)
		self.framespinbox.setMaximum(0)
		self.framespinbox.valueChanged.connect(APP_STATE.set_frame_index)
		APP_STATE.update_frame_index.connect(self.framespinbox.setValue)
		APP_STATE.update_frames_max.connect(lambda x: self.framespinbox.setMaximum(x-1))
		
		self.timeslider = QSlider(QtCore.Qt.Horizontal)
		self.timeslider.setMinimum(0)
		self.timeslider.setMaximum(0)
		self.timeslider.valueChanged.connect(APP_STATE.set_frame_index)
		APP_STATE.update_frame_index.connect(self.timeslider.setValue)
		APP_STATE.update_frames_max.connect(lambda x: self.timeslider.setMaximum(x-1))

		self.setLayout(QHBoxLayout())
		self.layout().addWidget(self.timeslider)
		self.layout().addWidget(self.framespinbox)
		self.layout().addWidget(self.prevbtn)
		self.layout().addWidget(self.nextbtn)
		self.layout().setContentsMargins(0, 0, 0, 0)


class Controls(QWidget):
	def __init__(self, parent: Optional[QWidget] = None) -> None:
		super().__init__(parent)

		self.showids = QCheckBox('Show IDs')
		self.showids.stateChanged.connect(APP_STATE.set_show_ids)
		self.showlin = QCheckBox('Show lineage relations')
		self.showlin.stateChanged.connect(APP_STATE.set_show_lineage_graph)
		self.time = Timeline()
		self.setLayout(QHBoxLayout())
		self.layout().addWidget(self.showids)
		self.layout().addWidget(self.showlin)
		self.layout().addWidget(self.time)
		self.layout().setContentsMargins(0, 0, 0, 0)


class Canvas(QWidget):
	def __init__(self, parent: Optional[QWidget] = None) -> None:
		super().__init__(parent)

		self.view = pg.GraphicsView()
		self.view.scene().sigMouseClicked.connect(self.handle_mouseclick)
		self.vb = pg.ViewBox()
		self.vb.setAspectLocked()
		self.view.setCentralItem(self.vb)

		self.img_segmentation = pg.ImageItem()
		self.img_microscopy = pg.ImageItem()
		self.img_budneck = pg.ImageItem()
		self.img_nucleus = pg.ImageItem()
		self.img_segmentation.setCompositionMode(QtGui.QPainter.CompositionMode.CompositionMode_Plus)
		self.img_microscopy.setCompositionMode(QtGui.QPainter.CompositionMode.CompositionMode_Plus)
		self.img_budneck.setCompositionMode(QtGui.QPainter.CompositionMode.CompositionMode_Plus)
		self.img_nucleus.setCompositionMode(QtGui.QPainter.CompositionMode.CompositionMode_Plus)
		self.img_segmentation_colormap = pg.ColorMap((0, 1), ('#000', '#00F'))
		self.img_microscopy_colormap = pg.ColorMap((0, 1), ('#000', '#FFF'))
		self.img_budneck_colormap = pg.ColorMap((0, 1), ('#000', '#0F0'))
		self.img_nucleus_colormap = pg.ColorMap((0, 1), ('#000', '#F00'))
		self.img_segmentation.setColorMap(self.img_segmentation_colormap)
		self.img_microscopy.setColorMap(self.img_microscopy_colormap)
		self.img_budneck.setColorMap(self.img_budneck_colormap)
		self.img_nucleus.setColorMap(self.img_nucleus_colormap)

		self.lineage_graph = pg.PlotCurveItem()

		self.hist_microscopy = pg.HistogramLUTWidget()
		self.hist_microscopy.setImageItem(self.img_microscopy)
		self.hist_microscopy.gradient.setColorMap(self.img_microscopy_colormap)
		self.hist_budneck = pg.HistogramLUTWidget()
		self.hist_budneck.setImageItem(self.img_budneck)
		self.hist_budneck.gradient.setColorMap(self.img_budneck_colormap)
		self.hist_nucleus = pg.HistogramLUTWidget()
		self.hist_nucleus.setImageItem(self.img_nucleus)
		self.hist_nucleus.gradient.setColorMap(self.img_nucleus_colormap)

		self.text_cellids: List[pg.TextItem] = []

		self.vb.addItem(self.img_microscopy)
		self.vb.addItem(self.img_budneck)
		self.vb.addItem(self.img_nucleus)
		self.vb.addItem(self.img_segmentation)
		self.vb.addItem(self.lineage_graph)

		self.setLayout(QGridLayout())
		self.layout().setSpacing(0)
		self.layout().addWidget(self.view, 0, 0)
		self.layout().addWidget(self.hist_microscopy, 0, 1)
		self.layout().addWidget(self.hist_budneck, 0, 2)
		self.layout().addWidget(self.hist_nucleus, 0, 3)
		self.layout().setContentsMargins(0, 0, 0, 0)

		APP_STATE.update_segmentation_data.connect(self.update_segmentation)
		APP_STATE.update_microscopy_data.connect(self.update_microscopy)
		APP_STATE.update_budneck_data.connect(self.update_budneck)
		APP_STATE.update_nucleus_data.connect(self.update_nucleus)
		APP_STATE.update_segmentation_data.connect(self.update_text_cellids)
		APP_STATE.update_segmentation_opacity.connect(lambda opacity: self.img_segmentation.setOpacity(opacity))
		APP_STATE.update_microscopy_opacity.connect(lambda opacity: self.img_microscopy.setOpacity(opacity))
		APP_STATE.update_budneck_opacity.connect(lambda opacity: self.img_budneck.setOpacity(opacity))
		APP_STATE.update_nucleus_opacity.connect(lambda opacity: self.img_nucleus.setOpacity(opacity))
		APP_STATE.update_show_ids.connect(self.update_text_cellids)
		APP_STATE.update_frame_index.connect(self.update_segmentation)
		APP_STATE.update_frame_index.connect(self.update_microscopy)
		APP_STATE.update_frame_index.connect(self.update_budneck)
		APP_STATE.update_frame_index.connect(self.update_nucleus)
		APP_STATE.update_frame_index.connect(self.update_text_cellids)
		APP_STATE.update_current_lineage_data.connect(self.update_lineage_graph)
		APP_STATE.update_segmentation_data.connect(self.update_lineage_graph)
		APP_STATE.update_frame_index.connect(self.update_lineage_graph)
		APP_STATE.update_show_lineage_graph.connect(self.update_lineage_graph)
		APP_STATE.update_centered_cellid.connect(self.update_centered_cellid)

	@Slot()
	def update_segmentation(self):
		if APP_STATE.data.segmentation is None:
			return

		self.img_segmentation.setImage(APP_STATE.data.segmentation.data[APP_STATE.values.frame_index], levels=(0, 1))

	@Slot()
	def update_microscopy(self):
		if APP_STATE.data.microscopy is None:
			return

		self.img_microscopy.setImage(APP_STATE.data.microscopy.data[APP_STATE.values.frame_index])

	@Slot()
	def update_budneck(self):
		if APP_STATE.data.budneck is None:
			return

		self.img_budneck.setImage(APP_STATE.data.budneck.data[APP_STATE.values.frame_index])

	@Slot()
	def update_nucleus(self):
		if APP_STATE.data.nucleus is None:
			return

		self.img_nucleus.setImage(APP_STATE.data.nucleus.data[APP_STATE.values.frame_index])

	@Slot()
	def update_text_cellids(self):
		if APP_STATE.data.segmentation is None:
			return

		# if not showing ids, just update visibility and quit
		if not APP_STATE.values.show_ids:
			for textitem in self.text_cellids:
				textitem.setVisible(APP_STATE.values.show_ids)
			return

		cellids = APP_STATE.data.segmentation.cell_ids(APP_STATE.values.frame_index)
		cms = APP_STATE.data.segmentation.cms(APP_STATE.values.frame_index)

		# remove unused text items
		while len(self.text_cellids) > len(cellids):
			item = self.text_cellids.pop()
			self.vb.removeItem(item)

		# add new text items as needed
		while len(self.text_cellids) < len(cellids):
			self.text_cellids.append(pg.TextItem(fill='#FFF8', color='#000', anchor=(0.5, 0.5)))
			self.vb.addItem(self.text_cellids[-1])

		# update labels and positions
		for textitem, cellid, cm in zip(self.text_cellids, cellids, cms):
			textitem.setText(f'{cellid:d}')
			textitem.setPos(cm[1], cm[0])
			textitem.setVisible(APP_STATE.values.show_ids)

	@Slot()
	def update_lineage_graph(self):
		if not APP_STATE.values.show_lineage_graph:
			self.lineage_graph.setOpacity(0)
			return
		else:
			self.lineage_graph.setOpacity(1)
		
		lineage = APP_STATE.data.current_lineage
		segmentation = APP_STATE.data.segmentation

		if lineage is None or segmentation is None:
			return

		mask = (lineage.time_ids <= APP_STATE.values.frame_index) & (lineage.parent_ids > 0) & (lineage.bud_ids > 0)
		xy = np.full((2, len(mask)*2*3), np.nan)  # 2 points per segment, 3 segments per budding event, len(mask) budding events

		a = np.pi/6  # angle of the arrow wings
		c = np.cos(a)
		s = np.sin(a)
		R = np.array(((c, -s), (s, c)))
		wing = 3  # length of the arrow wings
		radius = 3  # radius of the disk around the center of mass

		for idt, (parent_id, bud_id) in enumerate(zip(lineage.parent_ids[mask], lineage.bud_ids[mask])):
			cm_parent = segmentation.cms(APP_STATE.values.frame_index, [parent_id])[0]
			cm_bud = segmentation.cms(APP_STATE.values.frame_index, [bud_id])[0]

			vec = cm_bud - cm_parent
			length = np.sqrt(vec[0]**2 + vec[1]**2)
			vec_pad = vec * radius/length

			# padded end to end vector
			p1 = cm_parent + vec_pad
			p2 = cm_bud - vec_pad
			
			# arrow body
			xy[:, 6*idt] = p1
			xy[:, 6*idt+1] = p2

			# arrow wings
			p12 = p2 - p1
			p12 /= np.sqrt(p12[0]**2 + p12[1]**2)
			p12 *= wing

			wing_l = R @ p12
			wing_r = R.T @ p12
			xy[:, 6*idt+2] = p2
			xy[:, 6*idt+3] = p2 - wing_l
			xy[:, 6*idt+4] = p2
			xy[:, 6*idt+5] = p2 - wing_r

		self.lineage_graph.setData(xy[1], xy[0], connect='pairs')

	@Slot(int, int)
	def update_centered_cellid(self, timeid, cellid):
		if APP_STATE.data.segmentation is None:
			return

		center = APP_STATE.data.segmentation.cms(timeid, [cellid])[0]
		size = 100
		rect = (center[1]-size/2, center[0]-size/2, size, size)
		self.vb.setRange(QtCore.QRectF(*rect))

	@Slot(QtGui.QMouseEvent)
	def handle_mouseclick(self, ev: QtGui.QMouseEvent):
		point = ev.pos()
		
		if self.vb.sceneBoundingRect().contains(point):
			point_view = self.vb.mapSceneToView(point)
			idx = [int(point_view.y()), int(point_view.x())]
			
			if APP_STATE.data.segmentation is None:
				return

			if idx[0] < 0 or idx[1] < 0 or idx[0] >= APP_STATE.data.segmentation.data.shape[1] or idx[1] >= APP_STATE.data.segmentation.data.shape[2]:
				return

			clicked_cellid = APP_STATE.data.segmentation.data[APP_STATE.values.frame_index, idx[0], idx[1]]
			APP_STATE.set_clicked_cellid(clicked_cellid)


class Viewer(QWidget):
	def __init__(self, parent: Optional[QWidget] = None) -> None:
		super().__init__(parent)

		self.layers = Layers()
		self.controls = Controls()
		self.canvas = Canvas()

		self.setLayout(QGridLayout())
		self.layout().addWidget(self.layers, 0, 0)
		self.layout().addWidget(self.canvas, 0, 1)
		self.layout().addWidget(self.controls, 1, 0, 1, 0)