if __name__ == '__main__':
	from ._gui import App, APP_STATE
	from PySide6 import QtWidgets
	from bread.data import Segmentation, Microscopy, Lineage

	app = QtWidgets.QApplication([])

	window = App()
	
	# APP_STATE.set_segmentation_data(Segmentation.from_h5('docs/source/examples/data/colony003_segmentation.h5'))
	# APP_STATE.set_microscopy_data(Microscopy.from_tiff('docs/source/examples/data/colony003_microscopy.tif'))
	# APP_STATE.set_budneck_data(Microscopy.from_tiff('docs/source/examples/data/colony003_GFP.tif'))
	# APP_STATE.set_nucleus_data(Microscopy.from_tiff('docs/source/examples/data/colony003_mCherry.tif'))
	# from pathlib import Path
	# lineage_fp = Path('docs/source/examples/data/colony003_lineage (copy).csv')
	# APP_STATE.add_lineage_data(Lineage.from_csv(lineage_fp), lineage_fp)

	print(APP_STATE)

	window.show()
	window.setWindowTitle('bread GUI')
	window.resize(1480, 480)

	app.exec()