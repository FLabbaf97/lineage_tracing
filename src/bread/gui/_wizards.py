from abc import abstractmethod
from typing import Optional, List
import warnings
from PySide6.QtWidgets import QWidget, QMenuBar, QMainWindow, QFormLayout, QVBoxLayout, QLabel, QHBoxLayout, QGridLayout, QPushButton, QCheckBox, QSlider, QTableWidget, QTableWidgetItem, QTabWidget, QGroupBox, QSpinBox, QDoubleSpinBox, QFileDialog, QMessageBox, QWizard, QWizardPage, QProgressBar, QProgressDialog
from PySide6.QtCore import QObject, Signal, Slot, QThread
from bread.algo.lineage import *
from bread.data import Lineage
from _state import APP_STATE

class NameAndDescription(QWidget):
	def __init__(self, name: str, desc: str, parent: Optional[QWidget] = None) -> None:
		super().__init__(parent)

		self.label_name = QLabel(name, self)
		self.label_name.setWordWrap(True)
		self.label_desc = QLabel(desc, self)
		self.label_desc.setStyleSheet('font-size: 0.7em; color: #555; min-width: 300px;')
		self.label_desc.setWordWrap(True)

		self.setLayout(QVBoxLayout())
		self.layout().addWidget(self.label_name)
		self.layout().addWidget(self.label_desc)

		self.layout().setContentsMargins(0, 0, 0, 0)


class GuesserParams(QWizardPage):
	def __init__(self, parent: Optional[QWidget] = None) -> None:
		super().__init__(parent)

		self.setTitle('Algorithm parameters')

		self.nn_threshold = QDoubleSpinBox(self)
		self.nn_threshold.setMinimum(0)
		self.nn_threshold.setSingleStep(0.5)

		self.flexible_fn_threshold = QCheckBox(self)

		self.num_frames_refractory = QSpinBox(self)

		self.setLayout(QFormLayout())
		self.layout().addRow(
			NameAndDescription(
				'Nearest neighbour threshold',
				'Cell masks separated by less than this threshold are considered neighbors, by default 8.0'
			),
			self.nn_threshold
		)
		self.layout().addRow(
			NameAndDescription(
				'Flexible nearest neighbour threshold',
				'If no nearest neighbours are found within the given threshold, try to find the closest one, by default False'
			),
			self.flexible_fn_threshold
		)
		self.layout().addRow(
			NameAndDescription(
				'Refractory period',
				'After a parent cell has budded, exclude it from the parent pool in the next frames. It is recommended to set it to a low estimate, as high values will cause mistakes to propagate in time. A value of 0 corresponds to no refractory period. by default 0'
			),
			self.num_frames_refractory
		)

		self.registerField('nn_threshold', self.nn_threshold)
		self.registerField('flexible_fn_threshold', self.flexible_fn_threshold)
		self.registerField('num_frames_refractory', self.num_frames_refractory)

	def initializePage(self):
		self.nn_threshold.setValue(LineageGuesser.nn_threshold)
		self.flexible_fn_threshold.setChecked(LineageGuesser.flexible_nn_threshold)
		self.num_frames_refractory.setValue(LineageGuesser.num_frames_refractory)


class GuesserBudneckParams(GuesserParams):
	def __init__(self, parent: Optional[QWidget] = None) -> None:
		super().__init__(parent)

		self.kernel_N = QSpinBox(self)
		self.kernel_N.setMinimum(0)

		self.kernel_sigma = QDoubleSpinBox(self)
		self.kernel_N.setMinimum(0)
		self.kernel_sigma.setSingleStep(1)

		self.offset_frames = QSpinBox(self)
		self.offset_frames.setMinimum(0)

		self.num_frames = QSpinBox(self)
		self.num_frames.setMinimum(0)

		self.layout().addRow(
			NameAndDescription(
				'Smoothing kernel size',
				'Size of the gaussian smoothing kernel in pixels, larger means smoother intensity curves. by default 30'
			),
			self.kernel_N
		)
		self.layout().addRow(
			NameAndDescription(
				'Smoothing kernel sigma',
				'Number of standard deviations to consider for the smoothing kernel. by default 1'
			),
			self.kernel_sigma
		)
		self.layout().addRow(
			NameAndDescription(
				'Offset frames',
				'Wait this number of frames after bud appears before guessing parent. Useful if the GFP peak is often delayed. by default 0'
			),
			self.offset_frames
		)
		self.layout().addRow(
			NameAndDescription(
				'Number of frames',
				'Number of frames to watch the budneck marker channel for. The algorithm makes a guess for each frame, then predicts a parent by majority-vote policy. by default 5'
			),
			self.num_frames
		)

		self.registerField('kernel_N', self.kernel_N)
		self.registerField('kernel_sigma', self.kernel_sigma)
		self.registerField('offset_frames', self.offset_frames)
		self.registerField('num_frames', self.num_frames)

	def initializePage(self):
		super().initializePage()
		self.kernel_N.setValue(LineageGuesserBudLum.kernel_N)
		self.kernel_sigma.setValue(LineageGuesserBudLum.kernel_sigma)
		self.offset_frames.setValue(LineageGuesserBudLum.offset_frames)
		self.num_frames.setValue(LineageGuesserBudLum.num_frames)


class GuesserExpSpeedParams(GuesserParams):
	def __init__(self, parent: Optional[QWidget] = None) -> None:
		super().__init__(parent)

		self.num_frames = QSpinBox(self)
		self.num_frames.setMinimum(0)

		self.ignore_dist_nan = QCheckBox(self)

		self.bud_distance_max = QDoubleSpinBox(self)
		self.bud_distance_max.setSingleStep(0.5)

		self.layout().addRow(
			NameAndDescription(
				'Number of frames',
				'How many frames to consider to compute expansion velocity. At least 2 frames should be considered for good results. by default 5'
			),
			self.num_frames
		)
		self.layout().addRow(
			NameAndDescription(
				'Ignore nan distances',
				'In some cases the computed expansion distance encounters an error (candidate parent flushed away, invalid contour, etc.), then the computed distance is replaced by nan for the given frame. If this happens for many frames, the computed expansion speed might be nan. Enabling this parameter ignores candidates for which the computed expansion speed is nan, otherwise raises an error. by default True'
			),
			self.ignore_dist_nan
		)
		self.layout().addRow(
			NameAndDescription(
				'Max interface distance',
				'Maximal distance (in pixels) between points on the parent and bud contours to be considered as part of the "budding interface". by default 7'
			),
			self.bud_distance_max
		)

		self.registerField('num_frames', self.num_frames)
		self.registerField('ignore_dist_nan', self.ignore_dist_nan)
		self.registerField('bud_distance_max', self.bud_distance_max)

	def initializePage(self):
		super().initializePage()
		self.num_frames.setValue(LineageGuesserExpansionSpeed.num_frames)
		self.ignore_dist_nan.setChecked(LineageGuesserExpansionSpeed.ignore_dist_nan)
		self.bud_distance_max.setValue(LineageGuesserExpansionSpeed.bud_distance_max)


class GuesserMinThetaParams(GuesserParams):
	def __init__(self, parent: Optional[QWidget] = None) -> None:
		super().__init__(parent)

		self.offset_frames = QSpinBox(self)
		self.offset_frames.setMinimum(0)

		self.num_frames = QSpinBox(self)
		self.num_frames.setMinimum(0)

		self.layout().addRow(
			NameAndDescription(
				'Offset frames',
				'Wait this number of frames after bud appears before guessing parent. by default 0'
			),
			self.offset_frames
		)
		self.layout().addRow(
			NameAndDescription(
				'Number of frames',
				'Number of frames to make guesses for after the bud has appeared. The algorithm makes a guess for each frame, then predicts a parent by majority-vote policy. by default 5'
			),
			self.num_frames
		)

		self.registerField('offset_frames', self.offset_frames)
		self.registerField('num_frames', self.num_frames)

	def initializePage(self):
		super().initializePage()
		self.offset_frames.setValue(LineageGuesserMinTheta.offset_frames)
		self.num_frames.setValue(LineageGuesserMinTheta.num_frames)


class GuesserMinDistanceParams(GuesserParams):
	def __init__(self, parent: Optional[QWidget] = None) -> None:
		super().__init__(parent)

	def initializePage(self):
		super().initializePage()


class GuesserLaunch(QWizardPage):
	def __init__(self, parent: Optional[QWidget] = None) -> None:
		super().__init__(parent)

		self.setTitle('Run algorithm')

		self.progress = QProgressBar(self)
		self.worker: Optional[QThread] = None

		self.setLayout(QVBoxLayout())
		self.layout().addWidget(self.progress)

	@property
	def running(self):
		return self.worker is not None and self.worker.isRunning()

	@Slot()
	def launch(self):
		if self.running:
			warnings.warn('cannot launch a new guesser thread, another one is still running')
			return

		self.worker = GuesserWorker(self, self.guesser())
		self.worker.progress.connect(self.handle_progress)
		self.worker.result.connect(self.handle_result)
		self.worker.finished.connect(self.worker.deleteLater)
		self.worker.start()
		self.completeChanged.emit()

	@abstractmethod
	def guesser(self) -> LineageGuesser:
		raise NotImplementedError()

	@Slot(int, int)
	def handle_progress(self, i: int, N: int):
		self.progress.setMaximum(N)
		self.progress.setValue(i+1)

	@Slot(Lineage)
	def handle_result(self, lineage: Lineage):
		APP_STATE.add_lineage_data(lineage)
		self.completeChanged.emit()

	def isComplete(self) -> bool:
		return not self.running

	def initializePage(self):
		self.launch()


class GuesserBudneckLaunch(GuesserLaunch):
	def __init__(self, parent: Optional[QWidget] = None) -> None:
		super().__init__(parent)

	def guesser(self):
		segmentation = APP_STATE.data.segmentation
		budneck_img = APP_STATE.data.budneck

		return LineageGuesserBudLum(
			segmentation=segmentation,
			budneck_img=budneck_img,
			num_frames_refractory=self.field('num_frames_refractory'),
			nn_threshold=self.field('nn_threshold'),
			flexible_nn_threshold=self.field('flexible_fn_threshold'),
			kernel_N=self.field('kernel_N'),
			kernel_sigma=self.field('kernel_sigma'),
			offset_frames=self.field('offset_frames'),
			num_frames=self.field('num_frames')
		)


class GuesserExpSpeedLaunch(GuesserLaunch):
	def __init__(self, parent: Optional[QWidget] = None) -> None:
		super().__init__(parent)

	def guesser(self):
		segmentation = APP_STATE.data.segmentation

		return LineageGuesserExpansionSpeed(
			segmentation=segmentation,
			nn_threshold=self.field('nn_threshold'),
			num_frames_refractory=self.field('num_frames_refractory'),
			flexible_nn_threshold=self.field('flexible_fn_threshold'),
			ignore_dist_nan=self.field('ignore_dist_nan'),
			bud_distance_max=self.field('bud_distance_max'),
		)


class GuesserMinThetaLaunch(GuesserLaunch):
	def __init__(self, parent: Optional[QWidget] = None) -> None:
		super().__init__(parent)

	def guesser(self):
		segmentation = APP_STATE.data.segmentation

		return LineageGuesserMinTheta(
			segmentation=segmentation,
			nn_threshold=self.field('nn_threshold'),
			num_frames_refractory=self.field('num_frames_refractory'),
			flexible_nn_threshold=self.field('flexible_fn_threshold'),
			num_frames=self.field('num_frames'),
			offset_frames=self.field('offset_frames'),
		)


class GuesserMinDistanceLaunch(GuesserLaunch):
	def __init__(self, parent: Optional[QWidget] = None) -> None:
		super().__init__(parent)

	def guesser(self):
		segmentation = APP_STATE.data.segmentation

		return LineageGuesserMinDistance(
			segmentation=segmentation,
			nn_threshold=self.field('nn_threshold'),
			num_frames_refractory=self.field('num_frames_refractory'),
			flexible_nn_threshold=self.field('flexible_fn_threshold'),
		)


class GuesserWorker(QThread):
	result = Signal(Lineage)
	progress = Signal(int, int)

	def __init__(self, parent: Optional[QWidget], guesser: LineageGuesser):
		super().__init__(parent)
		self.guesser = guesser

	def run(self):
		lineage_guess = self.guesser.guess_lineage(progress_callback=self.progress.emit)
		self.result.emit(lineage_guess)


class GuesserWizard(QWizard):
	def __init__(self, which: str, parent: Optional[QWidget] = None) -> None:
		super().__init__(parent)

		# https://stackoverflow.com/questions/35187729/pyqt5-double-spin-box-returning-none-value
		self.setDefaultProperty('QDoubleSpinBox', 'value', 'valueChanged')

		self.setOption(QWizard.NoCancelButtonOnLastPage)
		self.setOption(QWizard.NoBackButtonOnLastPage)
		self.setOption(QWizard.NoBackButtonOnStartPage)

		if which == 'LineageGuesserBudLum':
			self.addPage(GuesserBudneckParams())
			self.addPage(GuesserBudneckLaunch())
		elif which == 'LineageGuesserExpansionSpeed':
			self.addPage(GuesserExpSpeedParams())
			self.addPage(GuesserExpSpeedLaunch())
		elif which == 'LineageGuesserMinTheta':
			self.addPage(GuesserMinThetaParams())
			self.addPage(GuesserMinThetaLaunch())
		elif which == 'LineageGuesserMinDistance':
			self.addPage(GuesserMinDistanceParams())
			self.addPage(GuesserMinDistanceLaunch())
		else:
			raise RuntimeError(f'unkown lineage guesser {which}')

		self.page(0).setButtonText(QWizard.NextButton, 'Launch')

		self.setWindowTitle(f'bread GUI - {which}')