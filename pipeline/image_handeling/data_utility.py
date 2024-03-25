from __future__ import annotations
from os import sep, mkdir, remove, PathLike
from os.path import isdir, join
from image_handeling.Experiment_Classes import Experiment
from typing import Iterable
import numpy as np
from tifffile import imread, imwrite

def load_stack(img_list: list[PathLike], channel_list: str | Iterable[str], frame_range: int | Iterable[int], return_2D: bool=False)-> np.ndarray:
    # Convert to list if string or int
    if isinstance(channel_list, str):
        channel_list = [channel_list]
    
    if isinstance(frame_range, int):
        frame_range = [frame_range]
    
    # Load/Reload stack. Expected shape of images tzxyc
    exp_list = []
    for chan in channel_list:
        chan_list = []
        for frame in frame_range:
            f_lst = []
            for img in img_list:
                # To be able to load either _f3digit.tif or _f4digit.tif
                ndigit = len(img.split(sep)[-1].split('_')[2][1:])
                if chan in img and img.__contains__(f'_f%0{ndigit}d'%(frame+1)):
                    f_lst.append(imread(img))
            chan_list.append(f_lst)
        exp_list.append(chan_list)
    
    # Process stack
    if len(channel_list)==1:
        stack = np.squeeze(np.stack(exp_list))
    else:
        stack = np.moveaxis(np.squeeze(np.stack(exp_list)), [0], [-1])

    # If stack is already 2D or want to load 3D
    if len(f_lst)==1 or not return_2D:
        return stack
    
    # if stack is time series, then z is axis 1
    if len(frame_range)>1:
        return np.amax(stack, axis=1)
    # if not then z is axis 0
    else:
        return np.amax(stack, axis=0)

def img_list_src(exp_set: Experiment, img_fold_src: str)-> list[PathLike]:
    """If not manually specified, return the latest processed images list"""
    
    if img_fold_src and img_fold_src == 'Images':
        return exp_set.raw_imgs_lst
    
    if img_fold_src and img_fold_src == 'Images_Registered':
        return exp_set.registered_imgs_lst
    
    if img_fold_src and img_fold_src == 'Images_Blured':
        return exp_set.blured_imgs_lst
    
    # If not manually specified, return the latest processed images list
    if exp_set.preprocess.img_blured:
        return exp_set.blured_imgs_lst
    elif exp_set.preprocess.frame_reg:
        return exp_set.registered_imgs_lst
    else:
        return exp_set.raw_imgs_lst

def mask_list_src(exp_set: Experiment, mask_fold_src: str, call_loca: str)-> list[PathLike]:
    """If not manually specified, return the latest processed images list"""
    
    if mask_fold_src == 'Masks_Threshold':
        return exp_set.threshold_masks_lst
    
    if mask_fold_src == 'Masks_Cellpose':
        return exp_set.cellpose_masks_lst
    
    if mask_fold_src == 'Masks_IoU_Track':
        return exp_set.iou_tracked_masks_lst
    
    if mask_fold_src == 'Masks_Manual_Track':
        return exp_set.man_tracked_masks_lst
    
    if mask_fold_src == 'Masks_GNN_Track':
        return exp_set.gnn_tracked_masks_lst
    
    # If not manually specified, return the latest processed images list
    if exp_set.masks.gnn_tracking:
        return exp_set.gnn_tracked_masks_lst
    elif exp_set.masks.manual_tracking:
        return exp_set.man_tracked_masks_lst
    elif exp_set.masks.iou_tracking:
        print("I'm here")
        return exp_set.iou_tracked_masks_lst
    elif exp_set.masks.cellpose_seg:
        return exp_set.cellpose_masks_lst
    else:
        return exp_set.threshold_masks_lst

# TODO: Add a check whether the images are in the save folder
def is_processed(process: dict, channel_seg: str, overwrite: bool)-> bool:
    if overwrite:
        return False
    if not process:
        return False
    if channel_seg not in process:
        return False
    return True

def create_save_folder(exp_path: PathLike, folder_name: str)-> PathLike:
    save_folder = join(sep,exp_path+sep,folder_name)
    if not isdir(save_folder):
        print(f" ---> Creating saving folder: {save_folder}")
        mkdir(save_folder)
        return save_folder
    print(f" ---> Saving folder already exists: {save_folder}")
    return save_folder

def delete_old_masks(class_setting_dict: dict, channel_seg: str, mask_files_list: list[PathLike], overwrite: bool=False)-> None:
    """Check if old masks exists, if the case, the delete old masks. Only
    if overwrite is True and class_setting_dict is not empty and channel_seg is in class_setting_dict"""
    if not overwrite:
        return
    if not class_setting_dict:
        return
    if channel_seg not in class_setting_dict:
        return
    print(f" ---> Deleting old masks for the '{channel_seg}' channel")
    files_list = [file for file in mask_files_list if file.__contains__(channel_seg)]
    for file in files_list:
        if file.endswith((".tif",".tiff",".npy")):
            remove(file)

def get_resolution(um_per_pixel: tuple[float,float])-> tuple[float,float]:
    x_umpixel,y_umpixel = um_per_pixel
    return 1/x_umpixel,1/y_umpixel

def save_tif(array: np.ndarray, save_path: PathLike, um_per_pixel: tuple[float,float], finterval: int)-> None:
    """Save array as tif with metadata"""
    imagej_metadata = {'finterval':finterval, 'unit': 'um'}
    imwrite(save_path,array.astype(np.uint16),imagej=True,metadata=imagej_metadata,resolution=get_resolution(um_per_pixel))

def gen_input_data(exp_set: Experiment, img_sorted_frames: dict[str,list], channels: str | list[str], **kwargs)-> list[dict]:
    """Generate input data for multi-processing or -threading. Add all additionnal arguments as kwargs

    Args:
        exp_set (Experiment): The experiment settings.
        img_sorted_frames (dict[str,list]): List of imgs sorted by frames.
        channels (list[str]): The list of channels to include.
        **kwargs: Additional keyword arguments.

    Returns:
        list[dict]: A list of dictionaries representing the input data.

    """
    input_data = [{**kwargs,
                   **{'imgs_path':img_sorted_frames[frame],
                   'frame':frame,
                   'channels':channels,
                   'metadata':{'um_per_pixel':exp_set.analysis.um_per_pixel,
                               'finterval':exp_set.analysis.interval_sec}}}
                  for frame in range(exp_set.img_properties.n_frames)]
    return input_data

