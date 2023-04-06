import numpy as np
from PySide6 import QtGui, QtWidgets, QtCore
from PySide6.QtWidgets import QWidget, QMenuBar, QMainWindow, QVBoxLayout, QLabel, QHBoxLayout, QGridLayout, QPushButton, QCheckBox, QSlider, QTableWidget, QTableWidgetItem, QTabWidget, QGroupBox, QSpinBox, QFileDialog, QMessageBox, QComboBox
from PySide6.QtGui import QIcon
from PySide6.QtCore import QObject, Signal, Slot
from typing import Optional, List
from pathlib import Path
import pyqtgraph as pg
from bread.data import Lineage, Microscopy, Segmentation
from _state import APP_STATE
from _utils import lerp
from _dialogs import FilleChannelMapperDialog, FileTypeDialog

__all__ = ['Viewer']

pg.setConfigOption('imageAxisOrder', 'row-major')

class OpenFile(QGroupBox):
	def __init__(self, parent: Optional[QWidget] = None, *args, **kwargs) -> None:
		super().__init__(parent, *args, **kwargs)

		self.title = QLabel('Open file')
		self.openbtnSeg = QPushButton('Open Segmentation')
		self.openbtnSeg.setIcon(QIcon(str(Path(__file__).parent / 'fugue-icons-3.5.6' / 'icons-shadowless' / 'folder-open-image.png')))
		self.openbtnSeg.clicked.connect(self.file_open_segmentation)
		self.openbtnSeg.setToolTip("Help: by clicking on the button, you can open a segmentation file (.h5).")

		self.openbtnMic = QPushButton('Add nd2 or tif files')
		self.openbtnMic.setIcon(QIcon(str(Path(__file__).parent / 'fugue-icons-3.5.6' / 'icons-shadowless' / 'folder-open-image.png')))
		self.openbtnMic.clicked.connect(self.file_open_microscopy)
		self.openbtnMic.setToolTip("Help: by clicking on the button, you can open a microscopy file (.nd2 or .tiff). \n after that, you can select that this file corresponds to which type of data (e.g. brightfield, nucleus, budnecks, etc.) \n or in case of nd2 files, you can select which channel corresponds to which type of data.")
		self.setLayout(QVBoxLayout())
		self.layout().addWidget(self.openbtnMic)
		self.layout().addWidget(self.openbtnSeg)
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
		filepath, filefilter = QFileDialog.getOpenFileName(self, 'Open microscopy', './', 'Microscopy files (*.tiff *.tif *.nd2)')
		filepath = Path(filepath)
		if filepath.is_dir():
			return  # user did not select anything

		if filepath.suffix in ['.tif', '.tiff']:
			microscopy = Microscopy.from_tiff(filepath)
			file_type_dialog = FileTypeDialog(self)
			result = file_type_dialog.exec()
			if result == file_type_dialog.Accepted:
				file_type = file_type_dialog.get_file_type()
				print(f"File type selected: {file_type}")
			else:
				print("File type dialog canceled")
			if(file_type == 'Brightfield/Phase Contrast'):
				APP_STATE.set_microscopy_data(microscopy)
			elif(file_type == 'Nucleus'):
				APP_STATE.set_nucleus_data(microscopy)
			elif(file_type == 'Budneck'):
				APP_STATE.set_budneck_data(microscopy)
			else:
				raise RuntimeError(f'Unsupported file type : {file_type}')
		
		elif filepath.suffix in ['.nd2']:
			microscopy_list, microscopy_channels = Microscopy.from_nd2(filepath)
			file_nd2_channel_dialog = FilleChannelMapperDialog(self, channels=microscopy_channels)
			result = file_nd2_channel_dialog.exec()
			if result == file_nd2_channel_dialog.Accepted:
				channel_result = file_nd2_channel_dialog.get_result()
			else:
				print("File type dialog canceled")
				channel_result = {}
			for channel in microscopy_channels:
				if channel in channel_result.keys():
					microscopy = microscopy_list[microscopy_channels.index(channel)]
					if(channel_result[channel] == 'Brightfield/Phase Contrast'):
						APP_STATE.set_microscopy_data(microscopy)
					elif(channel_result[channel] == 'Nucleus'):
						APP_STATE.set_nucleus_data(microscopy)
					elif(channel_result[channel] == 'Budneck'):
						APP_STATE.set_budneck_data(microscopy)

		else:
			raise RuntimeError(f'Unsupported extension : {filepath.suffix}')

class LayerConfig(QGroupBox):
	def __init__(self, title, parent: Optional[QWidget] = None, *args, **kwargs) -> None:
		super().__init__(parent, *args, **kwargs)

		self.opacityslider = QSlider(QtCore.Qt.Horizontal)
		self.opacityslider.setMinimum(0)
		self.opacityslider.setMaximum(10)
		self.opacityslider.setSingleStep(1)
		self.opacityslider.setValue(10)
		self.label = QLabel(str(title)+': ')
		self.lable_channel = QLabel(APP_STATE.data.get_coresponding_channel(title))
		self.file_name = QLabel(APP_STATE.data.get_coresponding_filepath(title))
		self.description = QWidget(self)
		self.description.setLayout(QHBoxLayout())
		self.description.layout().addWidget(self.label)
		self.description.layout().addWidget(self.lable_channel)
		self.setLayout(QVBoxLayout())
		self.layout().addWidget(self.description)
		self.layout().addWidget(self.file_name)
		self.layout().addWidget(self.opacityslider)

		self.layout().setContentsMargins(0, 0, 0, 0)


class LayerConfigs(QWidget):
	# TODO : time shape mismatch warning
	# TODO : shape checking for data
	
	def __init__(self, parent: Optional[QWidget] = None) -> None:
		super().__init__(parent)

		self.title = QLabel('Set Layers Opacity')
		self.segmentation = LayerConfig(title='Segmentation')
		self.segmentation.opacityslider.valueChanged.connect(lambda val: APP_STATE.set_opacity_segmentation(lerp(val, self.segmentation.opacityslider.minimum(), self.segmentation.opacityslider.maximum(), 0, 1)))

		self.microscopy = LayerConfig(title='Microscopy')
		self.microscopy.opacityslider.valueChanged.connect(lambda val: APP_STATE.set_opacity_microscopy(lerp(val, self.microscopy.opacityslider.minimum(), self.microscopy.opacityslider.maximum(), 0, 1)))

		self.budneck = LayerConfig(title='Budneck')
		self.budneck.opacityslider.valueChanged.connect(lambda val: APP_STATE.set_opacity_budneck(lerp(val, self.budneck.opacityslider.minimum(), self.segmentation.opacityslider.maximum(), 0, 1)))

		self.nucleus = LayerConfig(title='Nucleus')
		self.nucleus.opacityslider.valueChanged.connect(lambda val: APP_STATE.set_opacity_nucleus(lerp(val, self.nucleus.opacityslider.minimum(), self.segmentation.opacityslider.maximum(), 0, 1)))

		self.setLayout(QVBoxLayout())
		self.layout().setAlignment(QtCore.Qt.AlignTop)
		self.layout().setContentsMargins(0, 0, 0, 0)
		self.layout().addWidget(self.title)
		self.layout().addWidget(self.segmentation)
		self.layout().addWidget(self.microscopy)
		self.layout().addWidget(self.budneck)
		self.layout().addWidget(self.nucleus)
		self.setFixedWidth(200)
		self.layout().setContentsMargins(0, 0, 0, 0)

		APP_STATE.update_microscopy_data.connect(self.update_microscopy_info)
		APP_STATE.update_budneck_data.connect(self.update_budneck_info)
		APP_STATE.update_nucleus_data.connect(self.update_nucleus_info)
		APP_STATE.update_segmentation_data.connect(self.update_segmentation_info)

	@Slot()
	def update_microscopy_info(self):
		self.microscopy.lable_channel.setText(APP_STATE.data.get_coresponding_channel('Microscopy'))
		self.microscopy.file_name.setText(APP_STATE.data.get_coresponding_filepath('Microscopy'))
	@Slot()
	def update_budneck_info(self):
		self.budneck.lable_channel.setText(APP_STATE.data.get_coresponding_channel('Budneck'))
		self.budneck.file_name.setText(APP_STATE.data.get_coresponding_filepath('Budneck'))
	@Slot()
	def update_nucleus_info(self):
		self.nucleus.lable_channel.setText(APP_STATE.data.get_coresponding_channel('Nucleus'))
		self.nucleus.file_name.setText(APP_STATE.data.get_coresponding_filepath('Nucleus'))
	@Slot()
	def update_segmentation_info(self):
		self.segmentation.lable_channel.setText(APP_STATE.data.get_coresponding_channel('Segmentation'))
		self.segmentation.file_name.setText(APP_STATE.data.get_coresponding_filepath('Segmentation'))

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
		
		# FOV drop down menu
		self.button_fov = QComboBox()
		self.button_fov.addItems(APP_STATE.data.fov_list)
		self.button_fov.activated.connect(APP_STATE.set_fov)
		APP_STATE.update_fov_list.connect(self.reset_fov_list)

		self.time = Timeline()
		
		self.setLayout(QHBoxLayout())
		self.layout().addWidget(self.showids)
		self.layout().addWidget(self.showlin)
		self.layout().addWidget(self.button_fov)
		self.layout().addWidget(self.time)
		self.layout().setContentsMargins(0, 0, 0, 0)

	def reset_fov_list(self, fov_list: List[str]):
		self.button_fov.clear()
		self.button_fov.addItems(fov_list)

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
		APP_STATE.update_fov.connect(self.update_segmentation)
		APP_STATE.update_fov.connect(self.update_microscopy)
		APP_STATE.update_fov.connect(self.update_budneck)
		APP_STATE.update_fov.connect(self.update_nucleus)
		APP_STATE.update_fov.connect(self.update_text_cellids)
		APP_STATE.update_fov.connect(self.update_lineage_graph)

	@Slot()
	def update_segmentation(self):
		if APP_STATE.data.segmentation is None:
			return

		self.img_segmentation.setImage(APP_STATE.data.segmentation.get_frame(APP_STATE.values.fov,APP_STATE.values.frame_index))

	@Slot()
	def update_microscopy(self):
		if APP_STATE.data.microscopy is None:
			return

		self.img_microscopy.setImage(APP_STATE.data.microscopy.get_frame(APP_STATE.values.fov, APP_STATE.values.frame_index))

	@Slot()
	def update_budneck(self):
		if APP_STATE.data.budneck is None:
			return

		self.img_budneck.setImage(APP_STATE.data.budneck.get_frame(APP_STATE.values.fov, APP_STATE.values.frame_index))

	@Slot()
	def update_nucleus(self):
		if APP_STATE.data.nucleus is None:
			return

		self.img_nucleus.setImage(APP_STATE.data.nucleus.get_frame(APP_STATE.values.fov, APP_STATE.values.frame_index))

	@Slot()
	def update_text_cellids(self):
		if APP_STATE.data.segmentation is None:
			return

		# if not showing ids, just update visibility and quit
		if not APP_STATE.values.show_ids:
			for textitem in self.text_cellids:
				textitem.setVisible(APP_STATE.values.show_ids)
			return

		cellids = APP_STATE.data.segmentation.cell_ids(APP_STATE.values.fov, APP_STATE.values.frame_index)
		cms = APP_STATE.data.segmentation.cms(APP_STATE.values.fov,APP_STATE.values.frame_index, cell_ids=cellids)

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
			cm_parent = segmentation.cms(APP_STATE.values.fov, APP_STATE.values.frame_index, [parent_id])[0]
			cm_bud = segmentation.cms(APP_STATE.values.fov, APP_STATE.values.frame_index, [bud_id])[0]

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
	def update_centered_cellid(self, fov, timeid, cellid):
		if APP_STATE.data.segmentation is None:
			return

		center = APP_STATE.data.segmentation.cms(fov, timeid, [cellid])[0]
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

			if idx[0] < 0 or idx[1] < 0 or idx[0] >= (APP_STATE.data.segmentation.data[APP_STATE.values.fov]).shape[1] or idx[1] >= (APP_STATE.data.segmentation.data[APP_STATE.values.fov]).shape[2]:
				return

			clicked_cellid = APP_STATE.data.segmentation.data[APP_STATE.values.fov][APP_STATE.values.frame_index, idx[0], idx[1]]
			APP_STATE.set_clicked_cellid(clicked_cellid)


class Viewer(QWidget):
	def __init__(self, parent: Optional[QWidget] = None) -> None:
		super().__init__(parent)

		self.open_files = OpenFile()
		self.config_layers = LayerConfigs()
		self.controls = Controls()
		self.canvas = Canvas()

		self.setLayout(QGridLayout())
		self.layout().addWidget(self.open_files, 0, 0)
		self.layout().addWidget(self.config_layers, 1, 0)
		self.layout().addWidget(self.canvas, 0, 1, 2, 1)
		self.layout().addWidget(self.controls, 2, 0, 1, 2)