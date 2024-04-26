from __future__ import annotations
from os import PathLike
from os.path import join, getsize
from preprocess_utils import save_tif, create_save_folder
from metadata import get_metadata

from nd2 import ND2File
from tifffile import imread
import numpy as np
from concurrent.futures import ThreadPoolExecutor


################################## main function ###################################
def process_img(img_path: PathLike, active_channel_list: list[str] = [], full_channel_list: list[str] = [])-> dict:
    """Determine which process fct to use depending on the file type and size of the img"""
    # Get file metadata
    meta_dict = get_metadata(img_path, active_channel_list, full_channel_list)
    
    # If nd2 file is bigger than 20 GB, then process the ND2File obj frame by frame
    if getsize(img_path) > 20e9 and meta_dict['file_type'] == '.nd2':
        process_img_obj(meta_dict)
        return meta_dict
    
    # Else get the array (including nd2 and tiff) and process it
    if meta_dict['file_type'] == '.tif':
        array = imread(img_path)
    else:
        with ND2File(img_path) as nd_obj:
            array = nd_obj.asarray()
    # Process the array
    process_img_array(array,meta_dict)
    return meta_dict
    
def process_img_array(array: np.ndarray, meta_dict: dict)-> None:
    """Get an ndarray of the image stack and extract each image to be saved as tif file.
    It uses multithreading to save each image."""
    def write_array(img_param: tuple)-> None:
        """Nested function to write the image to the save path within multithreading."""
        # Unpack input data
        chan, array_slice = img_param
        frame,serie,z,_ = array_slice
        img_name = chan+'_s%02d'%(serie+1)+'_f%04d'%(frame+1)+'_z%04d'%(z+1)+'.tif'
        # Create save path and write the image
        parent_folder = save_dirs[serie]
        save_path = join(parent_folder,img_name)
        save_tif(array[array_slice],save_path,meta_dict['um_per_pixel'],meta_dict['interval_sec'])
    
    # Create all the params of the images
    img_params_lst = get_img_params_lst(meta_dict)
    # Adjust array with missing dimension
    array = expand_array_dim(array,meta_dict['axes'])
    # Create the save dir
    save_dirs = [create_save_folder(exp_path, 'Images') for exp_path in meta_dict['exp_path_list']]    
    # Write the images
    with ThreadPoolExecutor() as executor:
        executor.map(write_array,img_params_lst)
      
def process_img_obj(meta_dict: dict)-> None:
    """Get an ND2File obj map to extract each image to be saved as tif file.
    It uses multithreading to save each image."""
    def write_nd2_obj(img_param: tuple)-> None:
        # Unpack input data
        chan, array_slice = img_param
        frame,serie,z,chan_idx = array_slice
        img_name = chan+'_s%02d'%(serie+1)+'_f%04d'%(frame+1)+'_z%04d'%(z+1)+'.tif'
        # Get the image
        index = get_frame(nd_obj,frame,serie,z)
        img = nd_obj.read_frame(index)
        # Create save path and write the image
        parent_folder = save_dirs[serie]
        save_path = join(parent_folder,img_name)
        save_tif(img[chan_idx],save_path,meta_dict['um_per_pixel'],meta_dict['interval_sec'])
    
    # Create all the names for the images+metadata
    img_params_lst = get_img_params_lst(meta_dict)
    # Create the save dir
    save_dirs = [create_save_folder(exp_path, 'Images') for exp_path in meta_dict['exp_path_list']]
    # Generate input_data
    nd_obj = ND2File(meta_dict['img_path'])
    # Run multi-processing
    with ThreadPoolExecutor() as executor:
        executor.map(write_nd2_obj,img_params_lst)
    nd_obj.close()

def get_img_params_lst(meta_dict: dict)-> list[tuple]:
    """Return a list of tuples with the channel name and the array slice of the image."""
    # Create a name for each image
    img_params_lst = []
    for serie in range(meta_dict['n_series']):
        for f in range(meta_dict['n_frames']):
            for z in range(meta_dict['n_slices']):
                for chan in meta_dict['active_channel_list']:
                    # img_name_list.append(chan+'_s%02d'%(serie+1)+'_f%04d'%(f+1)+'_z%04d'%(z+1))
                    chan_idx = meta_dict['full_channel_list'].index(chan)
                    img_params_lst.append((chan,(f,serie,z,chan_idx)))
    return img_params_lst

def expand_array_dim(array: np.ndarray, axes: str)-> np.ndarray:
    """Add missing dimension of the ndarray to have a final TPZCYX array shape. 
    P = position (serie), T = time, Z = z-slice, C = channel, Y = height, X = width"""
    # Open tif file
    ref_axes = 'TPZCYX'
    
    if len(axes)<len(ref_axes):
        missing_axes = [ref_axes.index(ax) for ax in ref_axes if ax not in axes]
        # Add missing axes
        for ax in missing_axes:
            array = np.expand_dims(array,axis=ax)
    return array

def get_frame(nd_obj: ND2File, frame:int, serie:int, z_slice:int)-> int:
    """Extract index of a specfic image from ND2File obj"""
    for entry in nd_obj.events():
        # Add missing axes
        if 'T Index' not in entry:
            entry['T Index'] = 0
        if 'P Index' not in entry:
            entry['P Index'] = 0
        if 'Z Index' not in entry:
            entry['Z Index'] = 0
        # Extract Index
        if entry['Z Index'] == z_slice and entry['T Index'] == frame and entry['P Index'] == serie:
            return entry['Index']


if __name__ == "__main__":
    # Test tif image
    # img_path = '/home/Test_images/tiff/Run2/c2z25t23v1_tif.tif'
    # Test nd2 image
    img_path = '/home/Test_images/nd2/Run4/c4z1t91v1.nd2'
    process_img(img_path)
