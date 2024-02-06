from PySide6 import QtGui, QtWidgets, QtCore
from PySide6.QtWidgets import QWidget, QMenuBar, QMainWindow, QVBoxLayout, QLabel, QHBoxLayout, QGridLayout, QPushButton, QCheckBox, QSlider, QTableWidget, QTableWidgetItem, QTabWidget, QGroupBox, QSpinBox, QFileDialog, QMessageBox
from PySide6.QtGui import QAction, QIcon, QCloseEvent
from PySide6.QtCore import QObject, Signal, Slot
from typing import Optional, List
from pathlib import Path
import warnings
import numpy as np
from bread.data import Lineage, Microscopy, Segmentation
from ._state import APP_STATE
from ._wizards import GuesserWizard

__all__ = ['Editor']

class RowControls(QWidget):
	def __init__(self, parent: Optional[QWidget] = None) -> None:
		super().__init__(parent)

		# IDEA : inline row controls
		# https://forum.qt.io/topic/93621/add-buttons-in-tablewidget-s-row/8
		self.delbtn = QPushButton('Del row')
		self.delbtn.setIcon(QIcon(str(Path(__file__).parent / 'fugue-icons-3.5.6' / 'icons-shadowless' / 'table-delete-row.png')))
		self.addbtn = QPushButton('Add row')
		self.addbtn.setIcon(QIcon(str(Path(__file__).parent / 'fugue-icons-3.5.6' / 'icons-shadowless' / 'table-insert-row.png')))
		self.moveupbtn = QPushButton('Mv up')
		self.moveupbtn.setIcon(QIcon(str(Path(__file__).parent / 'fugue-icons-3.5.6' / 'icons-shadowless' / 'arrow-090.png')))
		self.movedownbtn = QPushButton('Mv down')
		self.movedownbtn.setIcon(QIcon(str(Path(__file__).parent / 'fugue-icons-3.5.6' / 'icons-shadowless' / 'arrow-270.png')))
		
		self.setLayout(QHBoxLayout())
		self.layout().addWidget(self.moveupbtn)
		self.layout().addWidget(self.movedownbtn)
		self.layout().addWidget(self.addbtn)
		self.layout().addWidget(self.delbtn)

		self.layout().setContentsMargins(0, 0, 0, 0)


class EditorTab(QWidget):
	COLOR_ERR_PARSE = QtGui.QColor(0, 0, 0, 128)
	COLOR_ERR_TIMEID = QtGui.QColor(255, 0, 0, 128)
	COLOR_ERR_CELLID = QtGui.QColor(255, 0, 0, 128)
	COLOR_WARN_CELLID = QtGui.QColor(255, 64, 64, 128)
	COLOR_SPECIAL_CELLID = QtGui.QColor(0, 64, 255, 128)

	update_dirty = Signal(bool)

	def __init__(self, parent: Optional[QWidget] = None) -> None:
		super().__init__(parent)

		self.filepath: Optional[Path] = None
		self.name: Optional[str] = None
		self.dirty: bool = False

		self.rowcontrols = RowControls()
		self.rowcontrols.delbtn.clicked.connect(self.del_row)
		self.rowcontrols.addbtn.clicked.connect(self.add_row)
		self.rowcontrols.moveupbtn.clicked.connect(self.moveup_row)
		self.rowcontrols.movedownbtn.clicked.connect(self.movedown_row)

		self.table = QTableWidget(self)
		self.table.setColumnCount(3)
		self.table.setHorizontalHeaderLabels(['Parent', 'Bud', 'Time'])
		APP_STATE.update_segmentation_data.connect(self.validate_all)
		self.table.verticalHeader().setVisible(False)
		self.table.setSortingEnabled(False)
		self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)

		self.table.cellChanged.connect(self.validate_cell)
		self.table.cellChanged.connect(lambda irow, icol: self.set_dirty(True))
		self.table.cellClicked.connect(self.handle_cell_clicked)

		self.setLayout(QVBoxLayout())
		self.layout().addWidget(self.rowcontrols)
		self.layout().addWidget(self.table)

	@Slot()
	def set_dirty(self, dirty: bool):
		self.dirty = dirty
		# print(f'set dirty {dirty} on {self}')
		self.update_dirty.emit(self.dirty)

	@Slot()
	def del_row(self):
		self.table.blockSignals(True)  # block signals while we edit the table
		irow = self.table.currentRow()
		if irow == -1:
			# no row selected
			return

		self.table.removeRow(irow)		
		self.table.blockSignals(False)  # restore signals

	@Slot()
	def add_row(self):
		irow = self.table.currentRow() + 1
		self.table.insertRow(irow)

		self.table.setItem(irow, 0, QTableWidgetItem(''))
		self.table.setItem(irow, 1, QTableWidgetItem(''))
		self.table.setItem(irow, 2, QTableWidgetItem('{:d}'.format(APP_STATE.values.frame_index)))

	@Slot()
	def moveup_row(self):
		self.table.blockSignals(True)  # block signals while we edit the table
		irow = self.table.currentRow()
		icol0 = self.table.currentColumn()

		if irow == -1 or irow == 0:
			# no row selected or is the top row
			return

		# save current row and insert a blank
		row_items = [self.table.takeItem(irow, icol) for icol in range(self.table.columnCount())]
		
		# move above row into current row
		for icol in range(self.table.columnCount()):
			item = self.table.takeItem(irow-1, icol)
			self.table.setItem(irow, icol, item)
			# print(self.table.rowCount())

		# move the saved items into above row
		for icol, item in enumerate(row_items):
			self.table.setItem(irow-1, icol, item)

		self.table.setCurrentCell(irow-1, icol0)
		self.table.blockSignals(False)  # restore signals

	@Slot()
	def movedown_row(self):
		self.table.blockSignals(True)  # block signals while we edit the table
		irow = self.table.currentRow()
		
		if irow == -1 or irow == self.table.rowCount()-1:
			# no row selected or is the bottom row
			return

		self.table.setCurrentCell(irow+1, self.table.currentColumn())
		self.moveup_row()
		self.table.setCurrentCell(irow+1, self.table.currentColumn())
		self.table.blockSignals(False)  # restore signals

	@Slot()
	def validate_all(self) -> bool:
		valid = True
		for irow in range(self.table.rowCount()):
			valid &= self.validate_cell(irow, 0)
			valid &= self.validate_cell(irow, 1)
			valid &= self.validate_cell(irow, 2)
		return valid

	@Slot(int, int)
	def validate_cell(self, irow: int, icol: int) -> bool:
		# print(f'request validating cell {irow} {icol}')
		item = self.table.item(irow, icol)
		content = self.parse_cell(irow, icol)

		signalsBlocked_ = self.table.signalsBlocked()
		self.table.blockSignals(True)

		if content is None:
			# invalid number format
			item.setBackground(self.COLOR_ERR_PARSE)
			item.setToolTip('[ERROR] Non-integer value')
			self.table.blockSignals(signalsBlocked_)  # restore state
			return False

		if icol == 0 or icol == 1:
			# validate cell id
			timeid = self.parse_cell(irow, 2)

			if timeid is None:
				item.setBackground(self.COLOR_WARN_CELLID)
				item.setToolTip('[WARNING] Could not validate cell because time id is invalid')
				self.table.blockSignals(signalsBlocked_)  # restore state
				return False

			# see first if the cell has a special id
			is_special = True
			try:
				Lineage.SpecialParentIDs(content)
			except ValueError:
				is_special = False
				# cell id is not a special id
			
			if is_special:
				item.setBackground(self.COLOR_SPECIAL_CELLID)
				item.setToolTip(f'[INFO] Cell {content} is special ({Lineage.SpecialParentIDs(content).name})')
				self.table.blockSignals(signalsBlocked_)  # restore state
				return False

			if not APP_STATE.data.valid_cellid(cellid=content, timeid=timeid, fov=APP_STATE.values.fov):
				item.setBackground(self.COLOR_ERR_CELLID)
				item.setToolTip(f'[ERROR] Cell {content} does not exist at frame {timeid}')
				self.table.blockSignals(signalsBlocked_)  # restore state
				return False


		if icol == 2:
			# validate time id
			if not APP_STATE.data.valid_frameid(content):
				item.setBackground(self.COLOR_ERR_TIMEID)
				item.setToolTip('[ERROR] Frame index out of range')
				self.table.blockSignals(signalsBlocked_)  # restore state
				return False

			# cell validation depends on time validation
			self.validate_cell(irow, 0)
			self.validate_cell(irow, 1)

		# reset color if there is no error
		item.setBackground(QtGui.QBrush())
		item.setToolTip('')

		self.table.blockSignals(signalsBlocked_)  # restore state
		return True

	def open_lineage(self, lineage: Lineage, filepath: Optional[Path] = None):
		self.filepath = filepath
		nrows = len(lineage.time_ids)

		self.table.blockSignals(True)  # block signals while we edit the table

		self.table.clearContents()
		self.table.setRowCount(nrows)

		for irow, (parent_id, bud_id, time_id) in enumerate(zip(lineage.parent_ids, lineage.bud_ids, lineage.time_ids)):
			self.table.setItem(irow, 0, QTableWidgetItem('{:d}'.format(parent_id)))
			self.table.setItem(irow, 1, QTableWidgetItem('{:d}'.format(bud_id)))
			self.table.setItem(irow, 2, QTableWidgetItem('{:d}'.format(time_id)))
			# manually validate once the entire row is loaded
			self.validate_cell(irow, 0)
			self.validate_cell(irow, 1)
			self.validate_cell(irow, 2)

		self.table.blockSignals(False)  # restore signals

	def export_lineage(self):
		N: int = self.table.rowCount()
		parent_ids, bud_ids, time_ids = np.zeros(N, dtype=int), np.zeros(N, dtype=int), np.zeros(N, dtype=int)
		
		for irow in range(N):
			values = (self.parse_cell(irow, 0), self.parse_cell(irow, 1), self.parse_cell(irow, 2))
			if any(value is None for value in values):
				raise RuntimeError('lineage is not valid')
			parent_ids[irow] = values[0]
			bud_ids[irow] = values[1]
			time_ids[irow] = values[2]
		
		return Lineage(parent_ids, bud_ids, time_ids)

	@Slot(int, int)
	def handle_cell_clicked(self, irow: int, icol: int):
		def handle_col_time(irow, icol):
			timeid = self.parse_cell(irow, icol)
			if timeid is not None:
				APP_STATE.set_frame_index(timeid)

		def handle_col_cell(irow, icol):
			cellid = self.parse_cell(irow, icol)
			timeid = self.parse_cell(irow, 2)
			if timeid is not None:
				APP_STATE.set_frame_index(timeid)
			if cellid is not None and timeid is not None:
				APP_STATE.set_centered_cellid(timeid, cellid)
	
		if icol == 0 or icol == 1:
			# parent id or bud id
			handle_col_cell(irow, icol)
		elif icol == 2:
			# time index
			handle_col_time(irow, icol)

	def parse_cell(self, irow, icol) -> Optional[int]:
		contents = self.table.item(irow, icol).text()
		if contents == '':
			return None
		try:
			contents = int(contents)
		except ValueError as e:
			warnings.warn(f'cell at irow={irow}, icol={icol} contains non-digit data : {e}')
			return None
		return contents


class Editor(QWidget):
	def __init__(self, parent: Optional[QWidget] = None) -> None:
		super().__init__(parent)

		self.menubar = QMenuBar()
		self.menu_file = self.menubar.addMenu('&File')
		file_open_action = QAction('Open lineage', self)
		file_open_action.triggered.connect(self.file_open)
		self.menu_file.addAction(file_open_action)
		self.menu_file.addSeparator()
		file_save_action = QAction('Save lineage', self)
		file_save_action.triggered.connect(lambda *_, **__: self.file_save())  # discard the arguments passed
		self.menu_file.addAction(file_save_action)
		file_saveas_action = QAction('Save lineage as', self)
		file_saveas_action.triggered.connect(lambda *_, **__: self.file_saveas())  # discard the arguments passed
		self.menu_file.addAction(file_saveas_action)
		self.menu_file.addSeparator()
		file_close_action = QAction('Close current lineage', self)
		file_close_action.triggered.connect(lambda *_, **__: self.file_close())  # discard the arguments passed
		self.menu_file.addAction(file_close_action)
		self.menu_new = self.menubar.addMenu('&New')
		new_lineage_budneck = QAction('Guess lineage using budneck', self)
		new_lineage_budneck.triggered.connect(lambda: self.new_lineage_guesser('LineageGuesserBudLum'))
		self.menu_new.addAction(new_lineage_budneck)
		new_lineage_ML = QAction('Guess lineage using ML', self)
		new_lineage_ML.triggered.connect(lambda: self.new_lineage_guesser('LineageGuesserML'))
		self.menu_new.addAction(new_lineage_ML)
		new_lineage_expspeed = QAction('Guess lineage using expansion speed', self)
		new_lineage_expspeed.triggered.connect(lambda: self.new_lineage_guesser('LineageGuesserExpansionSpeed'))
		self.menu_new.addAction(new_lineage_expspeed)
		# new_lineage_mintheta = QAction('Guess lineage using min theta', self)
		# new_lineage_mintheta.triggered.connect(lambda: self.new_lineage_guesser('LineageGuesserMinTheta'))
		# self.menu_new.addAction(new_lineage_mintheta)
		# new_lineage_mindist = QAction('Guess lineage using min distance', self)
		# new_lineage_mindist.triggered.connect(lambda: self.new_lineage_guesser('LineageGuesserMinDistance'))
		# self.menu_new.addAction(new_lineage_mindist)
		self.menu_new.addSeparator()
		new_lineage_prefilled_action = QAction('Create pre-filled lineage file', self)
		new_lineage_prefilled_action.triggered.connect(self.new_lineage_prefilled)
		self.menu_new.addAction(new_lineage_prefilled_action)
		new_lineage_empty = QAction('Create empty lineage file', self)
		new_lineage_empty.triggered.connect(self.new_lineage_empty)
		self.menu_new.addAction(new_lineage_empty)
		# self.menu_vis = self.menubar.addMenu('&Visualize')
		# # MAYBE : implement this
		# self.menu_vis.addAction(QAction('Open graph view', self))

		self.editortabs = QTabWidget()
		APP_STATE.update_add_lineage_data.connect(self.add_lineage)

		self.setLayout(QVBoxLayout())
		self.layout().addWidget(self.menubar)
		self.layout().addWidget(self.editortabs)

		APP_STATE.closing.connect(self.autosave)

		self.setMaximumWidth(350)

	@Slot()
	def file_open(self):
		filepath, filefilter = QFileDialog.getOpenFileName(self, 'Open lineage', './', 'Lineage CSV files (*.csv)')
		filepath = Path(filepath)
		if filepath.is_dir():
			return  # user did not select anything
		lineage = Lineage.from_csv(filepath)
		APP_STATE.add_lineage_data(lineage, filepath)

	@Slot()
	def file_save(self, tab: Optional[EditorTab] = None):
		if tab is None:
			tab: EditorTab = self.editortabs.currentWidget()

		# no filename has been defined
		if tab.filepath is None:
			self.file_saveas()
		else:
			try:
				lineage = tab.export_lineage()
			except RuntimeError as e:
				QtWidgets.QMessageBox.warning(self, 'bread GUI warning', 'Could not save the linage, the table contains non-integer characters')
				return
			lineage.save_csv(tab.filepath)
			tab.set_dirty(False)
			APP_STATE.set_current_lineage_data(lineage)

	@Slot()
	def file_saveas(self, tab: Optional[EditorTab] = None):
		if tab is None:
			tab: EditorTab = self.editortabs.currentWidget()

		try:
			lineage = tab.export_lineage()
		except RuntimeError as e:
			QtWidgets.QMessageBox.warning(self, 'bread GUI warning', 'Could not save the linage, the table contains non-integer characters')
			return
		filepath, filefilter = QFileDialog.getSaveFileName(self, 'Save lineage', './', 'Lineage CSV files (*.csv)')
		filepath = Path(filepath)
		if filepath.is_dir():
			return  # user did not select anything
		lineage.save_csv(filepath)
		tab.filepath = filepath
		tab.name = filepath.name
		tab.set_dirty(False)
		APP_STATE.set_current_lineage_data(lineage)

	@Slot()
	def file_close(self):
		tabindex: int = self.editortabs.currentIndex()
		tab: EditorTab = self.editortabs.currentWidget()
		if tab.dirty:
			self.file_confirm_close(tab)
		self.editortabs.removeTab(tabindex)

	def file_confirm_close(self, tab: Optional[EditorTab] = None):
		if tab is None:
			tab: EditorTab = self.editortabs.currentWidget()

		msg = QMessageBox()
		msg.setText(f'The lineage `{tab.name}` has been modified.')
		msg.setInformativeText('Do you want to save your changes?')
		msg.setStandardButtons(QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
		msg.setDefaultButton(QMessageBox.Save)
		msg.setWindowTitle('bread GUI')
		ret = msg.exec()
		if ret == QMessageBox.Save:
			self.file_save(tab)
		elif ret == QMessageBox.Discard:
			pass
		elif ret == QMessageBox.Cancel:
			return

	@Slot()
	def new_lineage_empty(self):
		lineage = Lineage(np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int))
		APP_STATE.add_lineage_data(lineage)

	@Slot()
	def new_lineage_prefilled(self):
		if APP_STATE.data.segmentation is None:
			QMessageBox.warning(self, 'bread GUI warning', 'No segmentation loaded, unable to prefill new lineage file.\nCreate an empty lineage file instead, or load a segmentation.')
			return

		lineage = APP_STATE.data.segmentation.find_buds()
		APP_STATE.add_lineage_data(lineage)
	
	@Slot()
	def new_lineage_guesser(self, which: str):
		if which in ['LineageGuesserBudLum', 'LineageGuesserExpansionSpeed', 'LineageGuesserMinDistance', 'LineageGuesserMinTheta', 'LineageGuesserML'] and APP_STATE.data.segmentation is None:
			QMessageBox.warning(self, 'bread GUI warning', 'No segmentation loaded.\nCreate an empty lineage file instead, or load a segmentation.')
			return
		
		if which in ['LineageGuesserBudLum'] and APP_STATE.data.budneck is None:
			QMessageBox.warning(self, 'bread GUI warning', 'No budneck channel loaded.\nCreate an empty lineage file instead, or load a budneck channel.')
			return

		wizard = GuesserWizard(which, self)
		wizard.show()

	@Slot(Lineage, Path)
	def add_lineage(self, lineage: Lineage, filepath: Optional[Path]):
		editortab = EditorTab()
		editortab.open_lineage(lineage, filepath)
		editortab.name = filepath.name if filepath is not None else 'Unsaved lineage'
		editortab.update_dirty.connect(self.update_tab_label)
		self.editortabs.addTab(editortab, editortab.name)  # takes ownership
		self.editortabs.setCurrentWidget(editortab)
		if editortab.filepath is None:
			editortab.set_dirty(True)  # unsaved files are always dirty
		APP_STATE.set_current_lineage_data(lineage)

	@Slot()
	def update_tab_label(self):
		tab: EditorTab = self.editortabs.currentWidget()
		text: str = tab.name + (' (*)' if tab.dirty else '')
		self.editortabs.setTabText(self.editortabs.currentIndex(), text)

	@Slot()
	def autosave(self):
		for idx_tab in range(self.editortabs.count()):
			tab = self.editortabs.widget(idx_tab)
			if tab.dirty:
				self.file_confirm_close(tab)