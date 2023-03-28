import numpy as np
import os
from typing import Union, List
# `List` instead of `list` for backwards compatibility

__all__ = ['dump_npz', 'load_npz']

def dump_npz(filepath, data: dict, force=False, dry=False, compress=True):
	"""Dumps data to disk in .npz format

	Parameters
	----------
	filepath : str
		filepath of the .npz file
	data : dict
		data to dump. each item of the dict should be an array-like
	force : bool, optional
		allow overriding files. Defaults to False
	dry : bool, optional
		whether to actually write data. Defaults to False
	compress : bool, optional
		whether to compress the data. Defaults to True
	"""

	if not force and os.path.exists(filepath):
		if input(f'file {filepath} already exists. continue ? [y/n]') != 'y':
			return

	if dry:
		return

	os.makedirs(os.path.dirname(filepath), exist_ok=True)

	if compress:
		savefn = np.savez_compressed
	else:
		savefn = np.savez

	with open(filepath, 'wb') as file:  # if using a string path, numpy appends .npz
		savefn(file, **data)


def load_npz(filepath: Union[str, List[str]], key=None, autoflatten=True):
	if isinstance(filepath, list):
		return [load_npz(x, key) for x in filepath]

	dat = np.load(filepath)

	if key is None:
		if len(dat.files) == 1:
			return dat[dat.files[0]]
		else:
			return dict(dat)
	else:
		return dat[key]
