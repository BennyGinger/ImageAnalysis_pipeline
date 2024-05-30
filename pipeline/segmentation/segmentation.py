from __future__ import annotations
import json
from os import PathLike
from pathlib import Path
import cv2
from skimage.morphology import remove_small_objects, remove_small_holes
import numpy as np
from pipeline.segmentation.cp_segmentation import get_img_prop
from pipeline.image_handeling.data_utility import load_stack, create_save_folder, save_tif, run_multithread


############################# main functions ######################################
def threshold(img_paths: list[PathLike], channel_seg: str, overwrite: bool=False, manual_threshold: int=None, um_per_pixel: tuple[float,float]=None, finterval: int=None)-> dict:
    # Set up segmentation
    exp_path: Path = Path(img_paths[0]).parent.parent
    print(f" --> Segmenting object in {exp_path}")
    save_path: Path = Path(create_save_folder(exp_path,'Masks_Threshold'))
    
    # Check if exist
    if any(file.match(f"*{channel_seg}*") for file in save_path.glob('*.tif')) and not overwrite:
        # Log
        print(f"  ---> Object has already been segmented for the channel {channel_seg}")
        return load_metadata(exp_path,channel_seg)
    
    # Generate input data
    frames, _ = get_img_prop(img_paths)
    fixed_args = {'img_paths':img_paths,
                  'channel':channel_seg,
                  'manual_threshold':manual_threshold,
                  'process_as_2D':True,
                  'metadata':{'um_per_pixel':um_per_pixel,
                              'finterval':finterval}}
    
    log = f"Manual Threshold of {manual_threshold}" if manual_threshold else "Automatic Threshold"
    print(f"  ---> Segmenting object with {log}")
    
    # Apply threshold
    results = run_multithread(apply_threshold,range(frames),fixed_args)
    threshold_value_list = [thres_val for thres_val in results]

    return create_threshold_settings(manual_threshold,threshold_value_list,Path(img_paths[0]).parent.stem)


############################# helper functions ######################################
def determine_threshold(img: np.ndarray, manual_threshold: float=None)-> float:
    # Set the threshold's value. Either as input or automatically if thres==None
    threshold_value = manual_threshold
    if not manual_threshold:
        threshold_value,_ = cv2.threshold(img.astype(np.uint8),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return threshold_value

def clean_mask(mask: np.ndarray)-> np.ndarray:
    mask = remove_small_holes(mask.astype(bool),50)
    return remove_small_objects(mask,1000).astype(np.uint16)

def create_threshold_settings(manual_threshold: float | None, threshold_value_list: list, fold_src: str)-> dict:
    log_value = "MANUAL"
    threshold_value = manual_threshold
    if not manual_threshold:
        log_value = "AUTOMATIC"
        threshold_value = round(np.mean(threshold_value_list),ndigits=2)
    print(f"  ---> {log_value} threshold value of {threshold_value}")
    return {'method':log_value,'threshold':threshold_value,'fold_src':fold_src}
        
def apply_threshold(frame: int, img_paths: list[PathLike], channel: str, process_as_2D: bool, manual_threshold: float, metadata: dict)-> float:
    # Load the image
    img = load_stack(img_paths,channel,frame,process_as_2D)
    # Save directory
    path_name = [path for path in sorted(img_paths) if f'_f{frame+1:04d}' in path and channel in path][0]
    mask_path = path_name.replace('Images','Masks_Threshold').replace('_Registered','').replace('_Blured','')
    # Get the threshold value
    threshold_value = determine_threshold(img,manual_threshold)
    # Apply the threshold
    _,mask = cv2.threshold(img.astype(np.uint8),threshold_value,255,cv2.THRESH_BINARY)
    # Save
    save_tif(mask,mask_path,**metadata)
    return threshold_value

def load_metadata(exp_path: Path, channel_to_seg: str)-> dict:
    """Function to load the metadata from the json file if it exists. Experiment obj are saved as json files,
    which contains the metadata for the experiment. The metadata contains the settings for threshold."""
    
    # Check if metadata exists, if not return empty settings
    setting_path = exp_path.joinpath('exp_settings.json')
    if not setting_path.exists():
        print(f"  ---> No metadata found for the '{channel_to_seg}' channel.")
        return {}
    # Load metadata
    print(f"  ---> Loading metadata for the '{channel_to_seg}' channel.")    
    with open(setting_path,'r') as fp:
        meta = json.load(fp)
    return meta['segmentation']['threshold_seg'][channel_to_seg]



if __name__ == "__main__":
    # Test the thresholding function
    folder = Path("/home/Test_images/tiff/Run2/c2z25t23v1_tif_s1/Images_Registered")
    img_paths = [str(path) for path in folder.glob('*.tif')]
    threshold(img_paths,'RFP',True,manual_threshold=100)

