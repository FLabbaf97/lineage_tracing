from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QFormLayout, QPushButton, QVBoxLayout, QDialog, QLabel, QComboBox, QDialogButtonBox
microscopy_options = ['Brightfield/Phase Contrast', 'Nucleus', 'Budneck', 'Others']

class FileTypeDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select File Type")

        self.setLayout(QFormLayout())

        self.label = QLabel("Select the file type:")
        self.layout().addWidget(self.label)
        self.comboBox = QComboBox(self)
        for option in microscopy_options:
            self.comboBox.addItem(option)

        self.layout().addRow(self.label, self.comboBox)
        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.layout().addWidget(self.buttonBox)

    def get_file_type(self):
        return self.comboBox.currentText()

class FilleChannelMapperDialog(QDialog):
    def __init__(self, parent=None, channels: list=['Channel0']):
        super().__init__(parent)
        self.setWindowTitle("Map channels in your nd2 file to corresponding types")
        
        self.setLayout(QFormLayout())
        self.comboBoxes = []
        for label_text in microscopy_options:
            label = QLabel(label_text)
            comboBox = QComboBox(self)
            comboBox.addItem("None")
            for channel in channels:
                comboBox.addItem(channel)
            self.comboBoxes.append(comboBox)
            self.layout().addRow(label, comboBox)

        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.layout().addWidget(self.buttonBox)

    def get_result(self):
        result_dict = {}
        for i in range(len(microscopy_options)):
            if self.comboBoxes[i].currentText() != "None":
                result_dict[self.comboBoxes[i].currentText()] = microscopy_options[i]
        return result_dict
