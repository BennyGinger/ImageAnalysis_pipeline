from __future__ import annotations
from os import PathLike
import cv2
from skimage.morphology import remove_small_objects, remove_small_holes
import numpy as np
from tifffile import imsave
from concurrent.futures import ThreadPoolExecutor
from pipeline.image_handeling.Experiment_Classes import Experiment
from pipeline.image_handeling.data_utility import is_processed, load_stack, create_save_folder, img_list_src, delete_old_masks, save_tif, gen_input_data

def determine_threshold(img: np.ndarray, manual_threshold: float=None)-> float:
    # Set the threshold's value. Either as input or automatically if thres==None
    threshold_value = manual_threshold
    if not manual_threshold:
        threshold_value,_ = cv2.threshold(img.astype(np.uint8),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return threshold_value

def clean_mask(mask: np.ndarray)-> np.ndarray:
    mask = remove_small_holes(mask.astype(bool),50)
    return remove_small_objects(mask,1000).astype(np.uint16)

def create_threshold_settings(manual_threshold: float | None, threshold_value_list: list)-> dict:
    log_value = "MANUAL"
    threshold_value = manual_threshold
    if not manual_threshold:
        log_value = "AUTOMATIC"
        threshold_value = round(np.mean(threshold_value_list),ndigits=2)
    print(f"\t---> Threshold created with: {log_value} threshold of {threshold_value}")
    return {'method':log_value,'threshold':threshold_value}
        
def apply_threshold(img_dict: dict)-> float:
    
    img = load_stack(img_dict['imgs_path'],img_dict['channels'],[img_dict['frame']],return_2D=True)
    
    # Save directory
    savedir = img_dict['imgs_path'][0].replace("Images","Masks_Threshold").replace('_Registered','').replace('_Blured','')
    
    threshold_value = determine_threshold(img,img_dict['manual_threshold'])
    
    # Apply the threshold
    _,mask = cv2.threshold(img.astype(np.uint8),threshold_value,255,cv2.THRESH_BINARY)
    
    # Save
    save_tif(mask,savedir,**img_dict['metadata'])
    return threshold_value

# # # # # # # # main functions # # # # # # # # # 
def threshold(exp_obj_lst: list[Experiment], channel_seg: str, overwrite: bool=False, manual_threshold: int=None, img_fold_src: PathLike="")-> list[Experiment]:
    for exp_obj in exp_obj_lst:
        # Activate the branch
        exp_obj.segmentation.is_threshold_seg = True
        # Check if exist
        if is_processed(exp_obj.segmentation.threshold_seg,channel_seg,overwrite):
                # Log
            print(f" --> Object has already been segmented for the channel {list(exp_obj.segmentation.threshold_seg.keys())}")
            continue
        
        # Initialize input args and save folder
        create_save_folder(exp_obj.exp_path,'Masks_Threshold')
        delete_old_masks(exp_obj.segmentation.threshold_seg,channel_seg,exp_obj.threshold_masks_lst,overwrite)
        
        # Sort images by frames and channels
        imgs_list = [img for img in img_list_src(exp_obj,img_fold_src) if channel_seg in img]
        sorted_frames = {frame:[img for img in imgs_list if f"_f{frame+1:04d}" in img] for frame in range(exp_obj.img_properties.n_frames)}
        
        # Generate input data
        img_data = gen_input_data(exp_obj,sorted_frames,channel_seg,manual_threshold=manual_threshold)
        
        print(f" --> Segmenting object...")
        # Determine threshold value
        with ThreadPoolExecutor() as executor:
            results = executor.map(apply_threshold,img_data)
            threshold_value_list = [result for result in results]

        # log
        settings_dict = create_threshold_settings(manual_threshold,threshold_value_list)

        # Save settings
        exp_obj.segmentation.threshold_seg[channel_seg] = settings_dict
        exp_obj.save_as_json()    
    return exp_obj_lst  

