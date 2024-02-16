from __future__ import annotations
from os import sep, mkdir, remove, PathLike
from os.path import isdir, join, isfile
from image_handeling.Experiment_Classes import Experiment
from typing import Iterable
import numpy as np
from tifffile import imread 

def load_stack(img_list: list[PathLike], channel_list: Iterable[str], frame_range: Iterable[int])-> np.ndarray:
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
    if len(channel_list)==1:
        stack = np.squeeze(np.stack(exp_list))
    else:
        stack = np.moveaxis(np.squeeze(np.stack(exp_list)), [0], [-1])
    return stack

def img_list_src(exp_set: Experiment, img_fold_src: PathLike)-> list[PathLike]:
    """If not manually specified, return the latest processed images list"""
    
    if img_fold_src and img_fold_src == 'Images':
        return exp_set.processed_images_list
    
    if img_fold_src and img_fold_src == 'Images_Registered':
        return exp_set.register_images_list
    
    if img_fold_src and img_fold_src == 'Images_Blured':
        return exp_set.blur_images_list
    
    # If not manually specified, return the latest processed images list
    if exp_set.process.img_blured:
        return exp_set.blur_images_list
    elif exp_set.process.img_registered:
        return exp_set.register_images_list
    else:
        return exp_set.processed_images_list

def mask_list_src(exp_set: Experiment, mask_fold_src: PathLike)-> list[PathLike]:
    """If not manually specified, return the latest processed images list"""
    
    if mask_fold_src and mask_fold_src == 'Masks_Threshold' or mask_fold_src == 'threshold_seg':
        return exp_set.mask_threshold_list
    
    if mask_fold_src and mask_fold_src == 'Masks_Cellpose' or mask_fold_src == 'cellpose_seg':
        return exp_set.mask_cellpose_list
    
    if mask_fold_src and mask_fold_src == 'Masks_IoU_Track' or mask_fold_src == 'iou_tracking':
        return exp_set.mask_iou_track_list
    
    # If not manually specified, return the latest processed images list
    if exp_set.masks.iou_tracking:
        return exp_set.mask_iou_track_list
    elif exp_set.masks.cellpose_seg:
        return exp_set.mask_cellpose_list
    else:
        return exp_set.mask_threshold_list

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
        print(f" ---> Creating folder: {save_folder}")
        mkdir(save_folder)
        return save_folder
    print(f" ---> Saving in folder: {save_folder}")
    return save_folder

def gen_input_data(exp_set: Experiment, img_fold_src: PathLike, channel_seg_list: list, **kwargs)-> list[dict]:
    img_path_list = img_list_src(exp_set,img_fold_src)
    channel_seg = channel_seg_list[0]
    input_data = []
    for frame in range(exp_set.img_properties.n_frames):
        input_dict = {}
        imgs_path = [img for img in img_path_list if f"_f{frame+1:04d}" in img and channel_seg in img]
        input_dict['imgs_path'] = imgs_path
        input_dict['frame'] = frame
        input_dict['channel_seg_list'] = channel_seg_list
        input_dict.update(kwargs)
        input_data.append(input_dict)
    return input_data

def delete_old_masks(class_setting_dict: dict, channel_seg: str, mask_files_list: list[PathLike], overwrite: bool=False)-> None:
    if not overwrite:
        return
    if not class_setting_dict:
        return
    if channel_seg not in class_setting_dict:
        return
    print(f" ---> Deleting old masks for the '{channel_seg}' channel")
    files_list = [file for file in mask_files_list if file.__contains__(channel_seg)]
    for file in files_list:
        if isfile(file):
            remove(file)
