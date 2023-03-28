from PySide6 import QtGui, QtWidgets, QtCore
from PySide6.QtWidgets import QWidget, QMenuBar, QMainWindow, QVBoxLayout, QLabel, QHBoxLayout, QGridLayout, QPushButton, QCheckBox, QSlider, QTableWidget, QTableWidgetItem, QTabWidget, QGroupBox, QSpinBox, QFileDialog, QMessageBox
from PySide6.QtGui import QAction, QIcon
from PySide6.QtCore import QObject, Signal, Slot
from typing import Optional, List
from ._state import APP_STATE
from ._editor import Editor
from ._viewer import Viewer

__all__ = ['App']

class App(QMainWindow):
	def __init__(self, parent: Optional[QWidget] = None) -> None:
		super().__init__(parent)

		self.viewer = Viewer()
		self.editor = Editor()

		self.menu_app = self.menuBar().addMenu('Application')
		show_doc = QAction('Documentation', self)
		show_doc.triggered.connect(self.show_doc)
		self.menu_app.addAction(show_doc)
		show_about = QAction('About', self)
		show_about.triggered.connect(self.show_about)
		self.menu_app.addAction(show_about)
		self.menu_app.addSeparator()
		action_quit = QAction('Quit', self)
		action_quit.triggered.connect(self.quit)
		self.menu_app.addAction(action_quit)

		self.setCentralWidget(QWidget())
		self.centralWidget().setLayout(QHBoxLayout())
		self.centralWidget().layout().addWidget(self.viewer)
		self.centralWidget().layout().addWidget(self.editor)

		# self.setStyleSheet('border: 1px solid red;')

	def show_doc(self):
		QMessageBox.information(self, 'bread GUI - documentation', '<p>See documentation at <a href="https://ninivert.github.io/bread/">https://ninivert.github.io/bread/</a></p>')

	def show_about(self):
		QMessageBox.information(self, 'bread GUI - about', '''
			<h1>Bread GUI</h1>\
			<p>See source code at <a href="https://github.com/ninivert/bread">https://github.com/ninivert/bread</a>, and documentation <a href="https://ninivert.github.io/bread/">https://ninivert.github.io/bread/</a></p>
			<p>See paper : <a href="https://github.com/ninivert/bread/blob/main/paper/Automatic%20lineage%20construction%20from%20time-lapse%20microscopy%20images.pdf">Automatic lineage construction from time-lapse yeast cells images</a></p>
		''')

	def quit(self):
		self.close()

	def closeEvent(self, event: QtGui.QCloseEvent) -> None:
		APP_STATE.closing.emit()
		event.accept()