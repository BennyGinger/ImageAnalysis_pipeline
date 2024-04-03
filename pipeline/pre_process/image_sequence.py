from __future__ import annotations
from os import sep, scandir, PathLike
from os.path import join, exists, getsize
import warnings
from image_handeling.Experiment_Classes import init_from_dict, init_from_json, Experiment
from image_handeling.data_utility import create_save_folder, save_tif
from nd2 import ND2File
from tifffile import imread
import numpy as np
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from .metadata import get_metadata


<<<<<<< HEAD
# img_obj: object

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
    img_obj, meta,img_name = img_data
    # with ND2File(meta['img_path']) as img_obj:
    # img_obj = ND2File(meta['img_path'])
        # img_obj_temp = copy.deepcopy(img_obj)
    # img_obj.close()
    serie,frame,z_slice = [int(i[1:])-1 for i in img_name.split('_')[1:]]
    chan = meta['full_channel_list'].index(img_name.split('_')[0])
    
    # Get the image
    index = get_frame(nd_obj=img_obj, timestamp=frame, field_of_view=serie, z_stack_number=z_slice)
    img = img_obj.read_frame(index)
    # img_obj.close()

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
        warnings.filterwarnings("ignore", category=UserWarning) # unableling user warnings, because we have to reuse the nd2_object for the different cores
        img_obj = ND2File(meta_dict['img_path'])
        img_name_list = [(img_obj, meta_dict,x) for x in img_name_list]
        with ProcessPoolExecutor() as executor: # nd2 file are messed up with multithreading
            executor.map(write_ND2,img_name_list)
        img_obj.close()
        warnings.filterwarnings("default", category=UserWarning) # enableing the warnings again
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
=======
################################## main function ###################################
def get_image_sequence(img_path: PathLike, active_channel_list: list[str], full_channel_list: list[str]=[], overwrite: bool=False)-> list[Experiment]:
>>>>>>> origin/main2.0_dev
    """Create an image seq for individual image files (.nd2 or .tif), based on the number of field of view and return a list of Settings objects"""
    # Get metadata
    meta_dict = get_metadata(img_path,active_channel_list,full_channel_list)
    
    exp_obj_list = []
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
            exp_obj = init_exp_obj(exp_path,meta_dict)
            exp_obj_list.append(exp_obj)
            # No need to save the settings as they are already saved
            continue
        
        # If images are not processed, extract imseq and initialize exp_set object
        print(f" --> Extracting images and converting to image sequence")
        process_img(meta_dict)
        
        exp_obj = init_exp_obj(exp_path,meta_dict)
        exp_obj.save_as_json()
        exp_obj_list.append(exp_obj)
    return exp_obj_list

################################ Satelite functions ################################
def init_exp_obj(exp_path: PathLike, meta_dict: dict)-> Experiment:
    """Initialize Experiment object from json file if exists, else from the metadata dict. 
    Return the Experiment object."""
    
    if exists(join(sep,exp_path+sep,'exp_settings.json')):
        exp_obj = init_from_json(join(sep,exp_path+sep,'exp_settings.json'))
    else:
        meta_dict['exp_path'] = exp_path
        exp_obj = init_from_dict(meta_dict)
    return exp_obj

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
                    img_params_lst.append({'array_slice':(serie,f,z,chan_idx),
                                          'serie':serie,
                                          'img_name':chan+'_s%02d'%(serie+1)+'_f%04d'%(f+1)+'_z%04d'%(z+1)})
    return img_params_lst

def expand_array_dim(array: np.ndarray, axes: str)-> np.ndarray:
    """Add missing dimension of the ndarray to have a final PTZCYX array shape. 
    P = position (serie), T = time, Z = z-slice, C = channel, Y = height, X = width"""
    # Open tif file
    ref_axes = 'PTZCYX'
    
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


    
