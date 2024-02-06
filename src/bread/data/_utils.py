import numpy as np
import os
from typing import Union, List
# `List` instead of `list` for backwards compatibility
import h5py
import glob
from PIL import Image
import numpy as np
import os
from nd2reader import ND2Reader
from pathlib import Path
import tifffile



__all__ = ['dump_npz', 'load_npz', 'tif_to_h5', 'tif_to_stack', 'nd2_to_tif']

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


def tif_to_h5(mask_dir, h5_file_name, dataset_name='FOV0'):
    # Find all .tif mask files in the directory and sort them by name
    mask_files = glob.glob(mask_dir + "*.tif")
    # Open the first .tif file to get the shape of the mask
    mask_shape = np.array(Image.open(mask_files[0])).shape

    # # Create the HDF5 file and dataset
    with h5py.File(h5_file_name, "w") as f:
        # dataset = f.create_dataset(dataset_name, (len(mask_files),) + mask_shape, dtype=np.uint8)
		# t_labels is an array of labels for each frame in mask_files. Ex: ['T0', 'T1']
		# each labels shows T-th file in mask_files
        t_labels = ['T{}'.format(i) for i in range(len(mask_files))]
        dataset = file['/{}/{}'.format(dataset_name, t_labels)]

        # Iterate over all .tif mask files and store each mask in the HDF5 dataset
        for i, mask_file in enumerate(mask_files):
    #         mask = np.array(Image.open(mask_file))
    #         dataset[i] = mask

    #     # Close the file to save changes to disk
    #     f.close()

        
        	dataset[dataset_name][t_labels[i]] = mask_file.to_numpy()
    file.close()    
    print("Conversion complete!")

def tif_to_stack(tif_dir, stack_filename):
    # Get a list of all TIFF files in the directory
    tif_files = os.listdir(tif_dir)
    tif_files = [f for f in tif_files if f.endswith('.tif')]

    # Sort the file list based on the frame numbers
    tif_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

    # Load the first frame to get the image size
    img_size = np.array(Image.open(os.path.join(tif_dir, tif_files[0]))).shape
    stack_volume = np.zeros((len(tif_files),) + img_size, dtype=np.uint16)

    # Load each frame and add it to the stack
    for i, tif_file in enumerate(tif_files):
        img = np.array(Image.open(os.path.join(tif_dir, tif_file)))
        stack_volume[i, :, :] = img

    # Save the stacked volume to a new TIFF file
    with Image.fromarray(stack_volume.squeeze()) as stack_tif:
        stack_tif.save(stack_filename)

    print(f'Saved stack volume to {stack_filename}')

	
def nd2_to_tif(nd2_dir: str, tif_path: str, fov: int=0, channel: int=0):
	# read the nd2 file, and convert the fov to tif
	# nd2_dir: directory of the nd2 file
	# fov: field of view to extract
	# tif_path: path to save the extracted tif file

	with ND2Reader(nd2_dir) as images:
		sizec = images.sizes['c'] if 'c' in images.sizes  else 1		                    
		sizev = images.sizes['v'] if 'v' in images.sizes else 1
		try:
			channel_names = images.metadata['channels'] if images.metadata['channels'] else [f'Channel{n}' for n in range(sizec)]
		except KeyError:
			channel_names = [f'Channel{n}' for n in range(sizec)]
		
		images.bundle_axes = 'tyx'  # Set the order of axes for the output array
		if sizec > 1:
			images.default_coords['c'] = channel  # Set the channel to read
		if sizev > 1:
			images.default_coords['v'] = fov  # Set the field of view to read

		stacked_frames = np.stack([frame for frame in images[0]])
		# Save as a single TIFF file
		tifffile.imsave(Path(tif_path), stacked_frames)