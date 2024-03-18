from __future__ import annotations
from os import sep, scandir, PathLike
from os.path import join, exists
from image_handeling.Experiment_Classes import init_from_dict, init_from_json, Experiment
from image_handeling.data_utility import create_save_folder, save_tif
# from nd2reader import ND2Reader
from nd2 import ND2File
from tifffile import imread
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from .metadata import get_metadata


def name_img_list(meta_dict: dict)-> list[PathLike]:
    """Return a list of generated image names based on the metadata of the experiment"""
    # Create a name for each image
    img_name_list = []
    for serie in range(meta_dict['n_series']):
        for t in range(meta_dict['n_frames']):
            for z in range(meta_dict['n_slices']):
                for chan in meta_dict['active_channel_list']:
                    img_name_list.append(chan+'_s%02d'%(serie+1)+'_f%04d'%(t+1)+'_z%04d'%(z+1))
    return img_name_list

def get_frame(nd_obj, timestamp:int, field_of_view:int, z_stack_number:int):
        field_of_view = 0 if field_of_view is None else field_of_view
        timestamp = 0 if timestamp is None else timestamp
        z_stack_number = 0 if z_stack_number is None else z_stack_number
        
        for entry in nd_obj.events():
                if ('Z Index' not in entry or entry['Z Index'] == z_stack_number) and entry['T Index'] == timestamp and entry['P Index'] == field_of_view:
                        return entry['Index']
        return None

def write_ND2(img_data: list)-> None:
    # Unpack img_data
    meta,img_name = img_data
    img_obj = ND2File(meta['img_path'])
    serie,frame,z_slice = [int(i[1:])-1 for i in img_name.split('_')[1:]]
    chan = meta['full_channel_list'].index(img_name.split('_')[0])
    
    # Get the image
    img = img_obj.read_frame(get_frame(nd_obj=img_obj, timestamp=frame, field_of_view=serie, z_stack_number=z_slice)) 

    # Save
    im_folder = join(sep,meta['exp_path_list'][serie]+sep,'Images')
    save_tif(img[chan],join(sep,im_folder+sep,img_name)+".tif",meta['um_per_pixel'],meta['interval_sec'])
    
def expand_dim_tif(img_path: PathLike, axes: str)-> np.ndarray:
    """Adjust the dimension of the image to TZCYX"""
    # Open tif file
    img = imread(img_path)
    ref_axes = 'TZCYX'
    
    if len(axes)<len(ref_axes):
        missing_axes = [ref_axes.index(x) for x in ref_axes if x not in axes]
        # Add missing axes
        for ax in missing_axes:
            img = np.expand_dims(img,axis=ax)
    return img

def write_tif(img_data: list)-> None:
    # Unpack img_data
    meta,img_name,img = img_data
    _,frame,z_slice = [int(i[1:])-1 for i in img_name.split('_')[1:]]             
    chan = meta['full_channel_list'].index(img_name.split('_')[0])
    
    im_folder = join(sep,meta['exp_path_list'][0]+sep,'Images')
    save_tif(img[frame,z_slice,chan,...],join(sep,im_folder+sep,img_name)+".tif",meta['um_per_pixel'],meta['interval_sec'])
    
def write_img(meta_dict: dict)-> None:
    # Create all the names for the images+metadata
    img_name_list = name_img_list(meta_dict)
    
    if meta_dict['file_type'] == '.nd2':
        # Add metadata and img_obj to img_name_list
        img_name_list = [(meta_dict,x) for x in img_name_list]
        with ProcessPoolExecutor() as executor: # nd2 file are messed up with multithreading
            executor.map(write_ND2,img_name_list)
    elif meta_dict['file_type'] == '.tif':
        # Add metadata and img to img_name_list
        img_arr = expand_dim_tif(meta_dict['img_path'],meta_dict['axes'])
        img_name_list = [(meta_dict,x,img_arr) for x in img_name_list]
        with ThreadPoolExecutor() as executor:
            executor.map(write_tif,img_name_list)

def init_exp_settings(exp_path: PathLike, meta_dict: dict)-> Experiment:
    """Initialize Settings object from json file or metadata"""
    
    if exists(join(sep,exp_path+sep,'exp_settings.json')):
        exp_set = init_from_json(join(sep,exp_path+sep,'exp_settings.json'))
    else:
        meta_dict['exp_path'] = exp_path
        exp_set = init_from_dict(meta_dict)
    return exp_set

def img_seq_exp(img_path: PathLike, active_channel_list: list[str], full_channel_list: list[str]=[], overwrite: bool=False)-> list[Experiment]:
    """Create an image seq for individual image files (.nd2 or .tif), based on the number of field of view and return a list of Settings objects"""
    # Get metadata
    meta_dict = get_metadata(img_path,active_channel_list,full_channel_list)
    
    # If img are already processed
    exp_set_list = []
    for serie in range(meta_dict['n_series']):
        exp_path = meta_dict['exp_path_list'][serie]
        meta_dict['exp_path'] = exp_path
        print(f"--> Checking exp {exp_path} for image sequence")
        
        if exists(join(sep,exp_path+sep,'REMOVED_EXP.txt')):
            print(" --> Exp. has been removed")
            continue
        
        save_folder = create_save_folder(exp_path,'Images')
        
        if any(scandir(save_folder)) and not overwrite:
            print(f" --> Images have already been converted to image sequence")
            exp_set_list.append(init_exp_settings(exp_path,meta_dict))
            continue
        
        # If images are not processed
        print(f" --> Extracting images and converting to image sequence")
        write_img(meta_dict)
        
        exp_set = init_from_dict(meta_dict)
        exp_set.save_as_json()
        exp_set_list.append(exp_set)
    return exp_set_list
    
# # # # # # # main function # # # # # # #
def img_seq_all(img_path_list: list[PathLike], active_channel_list: list=[], 
                          full_channel_list: list=[], img_seq_overwrite: bool=False)-> list[Experiment]:
    """Process all the images files (.nd2 or .tif) found in parent_folder and return a list of Settings objects"""
    exp_set_list = []
    for img_path in img_path_list:
        exp_set_list.extend(img_seq_exp(img_path,active_channel_list,full_channel_list,img_seq_overwrite))
    return exp_set_list

