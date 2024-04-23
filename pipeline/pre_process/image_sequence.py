from __future__ import annotations
from os.path import join, getsize
import warnings
from image_handeling.data_utility import save_tif
from nd2 import ND2File
from tifffile import imread
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor


################################## main function ###################################
def process_img(meta_dict: dict)-> None:
    """Determine which process fct to use depending on the file type and size of the img"""
    # If image is tiff, then process the array
    if meta_dict['file_type'] == '.tif':
        # Get array
        array = imread(meta_dict['img_path'])
        process_img_array(array,meta_dict)
        return
    
    # Else process nd2 file
    # If img lower than 20 GB then process array
    if getsize(meta_dict['img_path']) < 20e9:
        # Get array
        with ND2File(meta_dict['img_path']) as nd_obj:
            array = nd_obj.asarray()
            nd_obj.close()
        process_img_array(array,meta_dict)
        return
    # If more than 20 GB
    process_img_obj(meta_dict)

def process_img_array(array: np.ndarray, meta_dict: dict)-> None:
    """Get an ndarray of the image stack and extract each image to be saved as tif file.
    It uses multithreading to save each image."""
    # Create all the names for the images+metadata
    img_params_lst = get_img_params_lst(meta_dict)
    
    # Adjust array with missing dimension
    array = expand_array_dim(array,meta_dict['axes'])
    # Generate input data: list[dict]
    input_data = [{**{'metadata':meta_dict,
                    'array':array},**param}
                    for param in img_params_lst]
    
    with ThreadPoolExecutor() as executor:
        executor.map(write_array,input_data)
      
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

def process_img(meta_dict: dict)-> None:
    """Determine which process fct to use depending on the file type and size of the img"""
    # If image is tiff, then process the array
    if meta_dict['file_type'] == '.tif':
        # Get array
        array = imread(meta_dict['img_path'])
        process_img_array(array,meta_dict)
        return
    
    # Else process nd2 file
    # If img lower than 20 GB then process array
    if getsize(meta_dict['img_path']) < 20e9:
        # Get array
        with ND2File(meta_dict['img_path']) as nd_obj:
            array = nd_obj.asarray()
            nd_obj.close()
        process_img_array(array,meta_dict)
        return
    # If more than 20 GB
    process_img_obj(meta_dict)

def process_img_array(array: np.ndarray, meta_dict: dict)-> None:
    """Get an ndarray of the image stack and extract each image to be saved as tif file.
    It uses multithreading to save each image."""
    # Create all the names for the images+metadata
    img_params_lst = get_img_params_lst(meta_dict)
    
    # Adjust array with missing dimension
    array = expand_array_dim(array,meta_dict['axes'])
    # Generate input data: list[dict]
    input_data = [{**{'metadata':meta_dict,
                    'array':array},**param}
                    for param in img_params_lst]
    
    with ThreadPoolExecutor() as executor:
        executor.map(write_array,input_data)
      
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
                    img_params_lst.append({'array_slice':(f,serie,z,chan_idx),
                                          'serie':serie,
                                          'img_name':chan+'_s%02d'%(serie+1)+'_f%04d'%(f+1)+'_z%04d'%(z+1)})
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

def write_array(input_data: dict)-> None:
    """Write the image to the save path within multithreading."""
    # Unpack input data
    meta = input_data['metadata']
    # Create save path and write the image
    save_path = join(meta['exp_path_list'][input_data['serie']],'Images',input_data['img_name']+".tif")
    save_tif(input_data['array'][input_data['array_slice']],save_path,meta['um_per_pixel'],meta['interval_sec'])

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


    
