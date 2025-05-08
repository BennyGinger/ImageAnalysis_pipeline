from __future__ import annotations
from os import PathLike, scandir
from os.path import join, getsize, exists
from pathlib import Path

from nd2 import ND2File
from tifffile import imread
import numpy as np

from utilities.data_utility import save_tif, create_save_folder, run_multithread
from image_extraction.metadata import get_metadata



################################## main function ###################################
def create_img_seq(img_path: PathLike, active_channel_list: list[str] = [], full_channel_list: list[str] = [], overwrite: bool = False)-> list[PathLike | dict]:
    # Get file metadata
    metadata = get_metadata(img_path, active_channel_list, full_channel_list)
    
    # Process each image series
    metadatas: list[PathLike | dict] = []
    for exp_path in metadata['exp_path_list']:
        print(f" --> Creating image sequence for \033[94m{exp_path}\033[0m")
        
        # If exp has been processed but removed
        if exists(join(exp_path,'REMOVED_EXP.txt')):
            print("  ---> Exp. has been removed")
            continue
        
        # Create the save folder
        save_folder = create_save_folder(exp_path,'Images')
        
        # If exp has already been ran, look for the exp_settings.json file as metadata
        if exists(join(exp_path,'exp_settings.json')):
            json_path = join(exp_path,'exp_settings.json')
            metadatas.append(json_path)
        else: # If exp has not been ran, create new metadata dict
            metadata['exp_path'] = exp_path
            metadatas.append(metadata.copy()) 
        
        # If img are already processed
        if any(scandir(save_folder)) and not overwrite:
            print(f"  ---> Images have already been converted to image sequence")
            continue
        
        # If images are not processed, extract imseq and initialize exp_set object
        print(f"  ---> Extracting images and converting to image sequence")
        process_img(metadata,save_folder)
    return metadatas

################################## helper functions ###################################
def process_img(meta_dict: dict, save_folder: PathLike)-> None:
    """Determine which process fct to use depending on the file type and size of the img, create the image sequence
    and return the metadata dict."""
    img_path = meta_dict['img_path']
    # If nd2 file is bigger than 20 GB, then process the ND2File obj frame by frame
    if getsize(img_path) > 20e9 and meta_dict['file_type'] == '.nd2':
        process_img_obj(meta_dict,save_folder)
        return
    
    # Else get the array (including nd2 and tiff) and process it
    if meta_dict['file_type'] == '.tif':
        array = imread(img_path)
    else:
        with ND2File(img_path) as nd_obj:
            array = nd_obj.asarray()
    # Process the array
    process_img_array(array,meta_dict,save_folder)
    return 

def get_img_params_lst(meta_dict: dict, save_folder: PathLike)-> list[tuple]:
    """Return a list of tuples with the channel name and the array slice of the image."""
    # Get the actual serie number from save_folder
    serie = int(Path(save_folder).parent.stem.split('_s')[-1]) - 1
    
    # Create a name for each image
    img_params_lst = []
    for f in range(meta_dict['n_frames']):
        for z in range(meta_dict['n_slices']):
            for chan in meta_dict['active_channel_list']:
                chan_idx = meta_dict['full_channel_list'].index(chan)
                img_params_lst.append((chan,(f,serie,z,chan_idx)))
    return img_params_lst


################################## tiff functions ###################################
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

def process_img_array(array: np.ndarray, meta_dict: dict, save_folder: PathLike)-> None:
    """Get an ndarray of the image stack and extract each image to be saved as tif file.
    It uses multithreading to save each image."""
    
    # Create all the params of the images
    img_params_lst = get_img_params_lst(meta_dict,save_folder)
    # Adjust array with missing dimension
    array = expand_array_dim(array,meta_dict['axes'])
    # Get the metadata
    fixed_args = {'array':array,
                  'save_folder':save_folder,
                  'metadata':{'um_per_pixel':meta_dict['um_per_pixel'],
                              'finterval':meta_dict['interval_sec']}}
    run_multithread(_write_array,img_params_lst,fixed_args)
   
def _write_array(img_param: tuple, array: np.ndarray, save_folder: PathLike, metadata: dict)-> None:
    """Function to write the image to the save path within multithreading."""
    # Unpack input data
    chan, array_slice = img_param
    frame,serie,z,_ = array_slice
    img_name = chan+'_s%02d'%(serie+1)+'_f%04d'%(frame+1)+'_z%04d'%(z+1)+'.tif'
    # Create save path and write the image
    save_path = join(save_folder,img_name)
    save_tif(array[array_slice],save_path,**metadata)
   
   
################################## nd2 functions ###################################   
def process_img_obj(meta_dict: dict, save_folder: PathLike)-> None:
    """Get an ND2File obj map to extract each image to be saved as tif file.
    It uses multithreading to save each image."""
    
    # Create all the names for the images+metadata
    img_params_lst = get_img_params_lst(meta_dict,save_folder)
    # Get the metadata
    with ND2File(meta_dict['img_path']) as nd_obj:
        fixed_args = {'nd_obj':nd_obj,
                    'save_folder':save_folder,
                    'metadata':{'um_per_pixel':meta_dict['um_per_pixel'],
                                'finterval':meta_dict['interval_sec']}}
        # Run multi-processing
        run_multithread(_write_nd2_obj,img_params_lst,fixed_args)

def _write_nd2_obj(img_param: tuple, nd_obj: ND2File, save_folder: PathLike, metadata: dict)-> None:
    # Unpack input data
    chan, array_slice = img_param
    frame,serie,z,chan_idx = array_slice
    img_name = chan+'_s%02d'%(serie+1)+'_f%04d'%(frame+1)+'_z%04d'%(z+1)+'.tif'
    # Get the image
    index = get_frame(nd_obj,frame,serie,z,)
    img = nd_obj.read_frame(index)
    # Create save path and write the image
    save_path = join(save_folder,img_name)
    if img.ndim == 3:
        save_tif(img[chan_idx],save_path,**metadata)
    else:
        save_tif(img,save_path,**metadata)

def get_frame(nd_obj: ND2File, frame:int, serie:int, z_slice:int, metadata: dict)-> int:
    """To extract the index of a specific image from ND2File obj, if run in multithreading, use the lock to limit access to the save function."""
    if 'lock' in metadata:
        with metadata['lock']:
            return _get_frame(nd_obj,frame,serie,z_slice)
    return _get_frame(nd_obj,frame,serie,z_slice)

def _get_frame(nd_obj: ND2File, frame:int, serie:int, z_slice:int)-> int:
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
    # img_path = '/home/Test_images/nd2/Run1/c1z25t25v1_nd2.nd2'
    # img_path = '/home/Test_images/nd2/Run2/c2z25t23v1_nd2.nd2'
    img_path = r'/home/Dia/Ca2+_Itga_analysis/iso/1382x1172_iso_15%laser@5min-MaxIP.nd2'
    fold_paths = create_img_seq(img_path,active_channel_list=['RFP','GFP'],full_channel_list=['RFP','GFP'],overwrite=True)
    # fold_paths = create_img_seq(img_path,active_channel_list=['GFP','RFP'],full_channel_list=['GFP','RFP'],overwrite=True)
