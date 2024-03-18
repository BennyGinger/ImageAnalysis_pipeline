from __future__ import annotations
from os import sep, scandir, PathLike
from os.path import join, exists
from image_handeling.Experiment_Classes import init_from_dict, init_from_json, Experiment
from image_handeling.data_utility import create_save_folder, save_tif
from nd2reader import ND2Reader
from tifffile import imread
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from .metadata import get_metadata


################################## main function ###################################
def get_image_sequence(img_path: PathLike, active_channel_list: list[str], full_channel_list: list[str]=[], overwrite: bool=False)-> list[Experiment]:
    """Create an image seq for individual image files (.nd2 or .tif), based on the number of field of view and return a list of Settings objects"""
    # Get metadata
    meta_dict = get_metadata(img_path,active_channel_list,full_channel_list)
    
    exp_set_list = []
    for serie in range(meta_dict['n_series']):
        exp_path = meta_dict['exp_path_list'][serie]
        meta_dict['exp_path'] = exp_path
        print(f"--> Checking exp {exp_path} for image sequence")
        
        # If exp has been processed but removed
        if exists(join(sep,exp_path+sep,'REMOVED_EXP.txt')):
            print(" --> Exp. has been removed")
            continue
        
        save_folder = create_save_folder(exp_path,'Images')
        
        # If img are already processed
        if any(scandir(save_folder)) and not overwrite:
            print(f" --> Images have already been converted to image sequence")
            exp_set = init_exp_settings(exp_path,meta_dict)
            exp_set_list.append(exp_set)
            # No need to save the settings as they are already saved
            continue
        
        # If images are not processed, extract imseq and initialize exp_set object
        print(f" --> Extracting images and converting to image sequence")
        write_img(meta_dict)
        
        exp_set = init_exp_settings(exp_path,meta_dict)
        exp_set.save_as_json()
        exp_set_list.append(exp_set)
    return exp_set_list

################################ Satelite functions ################################
def create_img_name_list(meta_dict: dict)-> list[PathLike]:
    """Return a list of generated image names based on the metadata of the experiment"""
    # Create a name for each image
    img_name_list = []
    for serie in range(meta_dict['n_series']):
        for f in range(meta_dict['n_frames']):
            for z in range(meta_dict['n_slices']):
                for chan in meta_dict['active_channel_list']:
                    img_name_list.append(chan+'_s%02d'%(serie+1)+'_f%04d'%(f+1)+'_z%04d'%(z+1))
    return img_name_list

def extract_image_params(img_name: str, full_channel_list: list[str])-> tuple[int,int,int,int]:
    """Extract the serie,frame,z_slice and channel from the image name. Return a tuple with the extracted parameters."""
    # Get the serie,frame,z_slice from the img_name
    serie,frame,z_slice = [int(i[1:])-1 for i in img_name.split('_')[1:]]
    # To get the original index of the channel, as active and full channel list may not be the same
    chan = full_channel_list.index(img_name.split('_')[0])
    return serie,frame,z_slice,chan

def write_ND2(input_data: dict)-> None:
    meta = input_data['metadata']
    # Get img parameters
    serie,frame,z_slice,chan = extract_image_params(input_data['img_name'],meta['full_channel_list'])
    # Open the ND2 file
    img_obj = ND2Reader(meta['img_path'])
    # Create save path
    save_path = join(meta['exp_path_list'][serie],'Images',input_data['img_name']+".tif")
    # Get the image       
    if meta['n_slices']>1: 
        img = img_obj.get_frame_2D(c=chan,t=frame,z=z_slice,x=meta['img_width'],y=meta['img_length'],v=serie)
        save_tif(img,save_path,meta['um_per_pixel'],meta['interval_sec'])
        return
    
    img = img_obj.get_frame_2D(c=chan,t=frame,x=meta['img_width'],y=meta['img_length'],v=serie)
    save_tif(img,save_path,meta['um_per_pixel'],meta['interval_sec'])
     
def expand_dim_tif(img_path: PathLike, axes: str)-> np.ndarray:
    """Adjust the dimension of the image to TZCYX, if any dimension is missing. 
    Return the image as a numpy array."""
    # Open tif file
    img = imread(img_path)
    ref_axes = 'TZCYX'
    
    if len(axes)<len(ref_axes):
        missing_axes = [ref_axes.index(ax) for ax in ref_axes if ax not in axes]
        # Add missing axes
        for ax in missing_axes:
            img = np.expand_dims(img,axis=ax)
    return img

def write_tif(input_data: dict)-> None:
    meta = input_data['metadata']
    # Get the frame,z_slice from the img_name, no serie as there is only one serie for tiff files
    _,frame,z_slice,chan = extract_image_params(input_data['img_name'],meta['full_channel_list'])
    # Create save path
    save_path = join(meta['exp_path_list'][0],'Images',input_data['img_name']+".tif")
    save_tif(input_data['img'][frame,z_slice,chan,...],save_path,meta['um_per_pixel'],meta['interval_sec'])
    
def write_img(meta_dict: dict)-> None:
    # Create all the names for the images+metadata
    img_names = create_img_name_list(meta_dict)
    
    if meta_dict['file_type'] == '.nd2':
        # Generate input data: list[dict]
        input_data = [{'metadata':meta_dict,
                       'img_name':name}
                      for name in img_names]
        
        with ProcessPoolExecutor() as executor: # nd2 file are messed up with multithreading
            executor.map(write_ND2,input_data)
    
    elif meta_dict['file_type'] == '.tif':
        # Get the image with the correct dimension
        img_arr = expand_dim_tif(meta_dict['img_path'],meta_dict['axes'])
        # Generate input data: list[dict]
        input_data = [{'metadata':meta_dict,
                       'img_name':name,
                       'img':img_arr}
                      for name in img_names]
        
        with ThreadPoolExecutor() as executor:
            executor.map(write_tif,input_data)

def init_exp_settings(exp_path: PathLike, meta_dict: dict)-> Experiment:
    """Initialize Experiment object from json file if exists, else from the metadata dict. 
    Return the Experiment object."""
    
    if exists(join(sep,exp_path+sep,'exp_settings.json')):
        exp_set = init_from_json(join(sep,exp_path+sep,'exp_settings.json'))
    else:
        meta_dict['exp_path'] = exp_path
        exp_set = init_from_dict(meta_dict)
    return exp_set


