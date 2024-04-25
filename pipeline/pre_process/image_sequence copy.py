from __future__ import annotations
from os import PathLike, mkdir
from os.path import join, getsize, exists
import warnings
from preprocess_utils import save_tif
from metadata import get_metadata

from nd2 import ND2File
from tifffile import imread
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor


################################## main function ###################################
def process_img(img_path: PathLike, active_channel_list: list[str] = [], full_channel_list: list[str] = [])-> None:
    """Determine which process fct to use depending on the file type and size of the img"""
    # Get file metadata
    meta_dict = get_metadata(img_path, active_channel_list, full_channel_list)
    
    # If image is tiff, then process the array
    if meta_dict['file_type'] == '.tif':
        # Get array
        array = imread(img_path)
        process_img_array(array,meta_dict)
        return
    
    # Else process nd2 file
    # If img lower than 20 GB then process array
    if getsize(img_path) < 20e9:
        # Get array
        with ND2File(img_path) as nd_obj:
            array = nd_obj.asarray()
        process_img_array(array,meta_dict)
        return
    # If more than 20 GB
    process_img_obj(meta_dict)

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
        parent_folder = join(meta_dict['exp_path_list'][serie], 'Images')
        if not exists(parent_folder):
            mkdir(parent_folder)
        save_path = join(parent_folder,img_name)
        save_tif(array[array_slice],save_path,meta_dict['um_per_pixel'],meta_dict['interval_sec'])
    
    # Create all the params of the images
    img_params_lst = get_img_params_lst(meta_dict)
    # Adjust array with missing dimension
    array = expand_array_dim(array,meta_dict['axes'])
    # Write the images
    with ThreadPoolExecutor() as executor:
        executor.map(write_array,img_params_lst)
      
def process_img_obj(meta_dict: dict)-> None:
    """Get an ND2File obj map to extract each image to be saved as tif file.
    It uses multithreading to save each image."""
    # Create all the names for the images+metadata
    img_params_lst = get_img_params_lst(meta_dict)
    
    # unableling user warnings, because we have to reuse the nd2_object for the different cores
    warnings.filterwarnings("ignore", category=UserWarning) 
    
    # Generate input_data
    img_obj = ND2File(meta_dict['img_path'])
    input_data = [(img_obj,meta_dict,param) for param in img_params_lst]
    
    # Run multi-processing
    with ProcessPoolExecutor() as executor: # nd2 file are messed up with multithreading
        executor.map(write_nd2_obj,input_data)
    img_obj.close()
    
    # enableing the warnings again
    warnings.filterwarnings("default", category=UserWarning)

def get_img_params_lst(meta_dict: dict)-> list[dict]:
    """Return a list of dict containing the slice indexes of all images,
    that is the serie (position), frame, z-slice and channel of each images, as well as the name of the image
    to be saved"""
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



def write_nd2_obj(img_data: list)-> None:
    # Unpack img_data
    img_obj,meta,param = img_data
    serie,f,z,chan_idx = param['array_slice']
    # Get the image
    index = get_frame(nd_obj=img_obj, frame=f, serie=serie, z_slice=z)
    img = img_obj.read_frame(index)
    # Save
    save_path = join(meta['exp_path_list'][param['serie']],'Images',param['img_name']+".tif")
    save_tif(img[chan_idx],save_path,meta['um_per_pixel'],meta['interval_sec'])

def get_frame(nd_obj: ND2File, frame:int, serie:int, z_slice:int)-> int:
    """Extract index of a specfic image from ND2File obj"""
    for entry in nd_obj.events():
        # Add missing axes
        if 'T Index' not in entry:
            entry['T index'] = 0
        if 'P Index' not in entry:
            entry['P Index'] = 0
        if 'Z Index' not in entry:
            entry['Z index'] = 0
        # Extract Index
        if entry['Z Index'] == z_slice and entry['T Index'] == frame and entry['P Index'] == serie:
            return entry['Index']


if __name__ == "__main__":
    img_path = '/home/Test_images/tiff/Run2/c2z25t23v1_tif.tif'
    
    process_img(img_path)
