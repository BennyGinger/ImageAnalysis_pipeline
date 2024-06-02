from __future__ import annotations
from os import sep, listdir, PathLike
from pathlib import Path
from typing import Iterable
import warnings

from tqdm import tqdm
warnings.filterwarnings("ignore", category=RuntimeWarning) 
from os.path import join
from pipeline.image_handeling.data_utility import load_stack, create_save_folder, run_multithread, get_img_prop, save_tif, is_channel_in_lst
from pipeline.mask_transformation.complete_track import complete_track
from cellpose.utils import stitch3D
from cellpose.metrics import _intersection_over_union
from scipy.stats import mode
from skimage.measure import regionprops_table
from skimage.segmentation import relabel_sequential
import numpy as np
from threading import Lock


############################ Main function ############################
def iou_tracking(img_paths: list[PathLike], channel_track: str, stitch_thres_percent: float=0.5, shape_thres_percent: float=0.9, overwrite: bool=False, mask_appear: int=5, copy_first_to_start: bool=True, copy_last_to_end: bool=True, **kwargs)-> None | ValueError:
    """Perform IoU (Intersection over Union) based cell tracking on a list of experiments.

    Args:
        exp_obj_lst (list[Experiment]): List of Experiment objects to perform tracking on.
        channel_seg (str): Channel name for segmentation.
        stitch_thres_percent (float, optional): Stitching threshold percentage. Defaults to 0.5. Higher values will result in more strict tracking (excluding more cells)
        shape_thres_percent (float, optional): Shape threshold percentage. Defaults to 0.9. Lower values will result in tracks with more differences in shape between frames.
        overwrite (bool, optional): Flag to overwrite existing tracking results. Defaults to False.
        mask_appear (int, optional): Number of times a mask should appear to be considered valid. Defaults to 5.
        copy_first_to_start (bool, optional): Flag to copy the first mask to the start. Defaults to True.
        copy_last_to_end (bool, optional): Flag to copy the last mask to the end. Defaults to True.
        kwargs: Additional keyword arguments, especially for metadata.
    
    Returns:
        list[Experiment]: List of Experiment objects with updated tracking information.
    """
    
    # Set up tracking
    frames, _ = get_img_prop(img_paths)
    exp_path: Path = Path(img_paths[0]).parent.parent
    print(f" --> Tracking cells in {exp_path}")
    save_path: Path = Path(create_save_folder(exp_path,'Masks_IoU_Track'))
    
    # Check if time sequence and if channel was segmented
    if frames == 1:
        print(f" --> '{exp_path}' is not a time sequence")
        return
    if not is_channel_in_lst(channel_track,img_paths):
        raise ValueError(f" --> Channel '{channel_track}' not found in the provided masks")
        
    # Already processed?
    if any(file.match(f"*{channel_track}*") for file in save_path.glob('*.tif')) and not overwrite:
        # Log
        print(f"  ---> Cells have already been tracked for the '{channel_track}' channel")
        return
    
    # Load masks and track images
    print(f"  ---> Tracking cells for the '{channel_track}' channel")
    mask_stack = load_stack(img_paths,channel_track,range(frames),True)
    
    # Track masks
    mask_stack = track_cells(mask_stack,stitch_thres_percent)
    
    # Check shape similarity to avoid false masks
    mask_stack = check_mask_similarity(mask_stack,shape_thres_percent)
    
    # Morph missing masks
    mask_stack = complete_track(mask_stack,mask_appear,copy_first_to_start,copy_last_to_end)
    
    # Re-assign the new value to the masks and obj. Previous step may have created dicontinuous masks
    print('  ---> Reassigning masks value')
    mask_stack,_,_ = relabel_sequential(mask_stack)
    
    # Save the masks
    mask_paths = [file for file in img_paths if file.__contains__('_z0001')]
    metadata = unpack_kwargs(kwargs)
    fixed_args = {'mask_stack':mask_stack,
                  'mask_paths':mask_paths,
                  'metadata':metadata}
    run_multithread(save_mask,range(frames),fixed_args)


############################ Helper functions ############################
def track_cells(masks: np.ndarray, stitch_threshold: float) -> np.ndarray:
    """
    Track cells in a sequence of masks. Using the Cellpose stitch3D function to stitch masks together 
    and then create a master mask to compare with each frame.

    Args:
        masks (np.ndarray): Array of masks representing cells in each frame.
        stitch_threshold (float): Threshold value for stitching masks together.

    Returns:
        np.ndarray: Master mask representing the stitched cells.

    """
    print('  ---> Tracking cells...')
    # basic stitching/tracking from Cellpose
    masks = stitch3D(masks, stitch_threshold)
    # Create mask with all the most common masks value
    master_mask = create_master_mask(masks)
    return master_mask_stitching(masks, master_mask, stitch_threshold)

def create_master_mask(masks: np.ndarray) -> np.ndarray:
    """
    Create a master mask by performing a 'mode' operation to get the most common value present in most t-frames per pixel. 
    Ignoring background by setting zero to nan. Therefore conversion to float is needed.

    Args:
        masks (np.ndarray): Input array of masks.

    Returns:
        np.ndarray: The master mask with all possible cells on one mask.
    """
    print('  ---> Creating master mask')
    rawmasks_ignorezero = masks.copy().astype(float)
    rawmasks_ignorezero[rawmasks_ignorezero == 0] = np.nan
    master_mask = mode(rawmasks_ignorezero, axis=0, keepdims=False, nan_policy='omit')[0]
    # Convert back to int
    return np.ma.getdata(master_mask).astype(int)

def master_mask_stitching(masks: np.ndarray, master_mask: np.ndarray, stitch_threshold: float) -> np.ndarray:
    """
    Second round of stitch/tracking by using a mastermask to compare with every frame.
    Stitch 2D masks into 3D volume with stitch_threshold on IOU.
    Slightly changed code from Cellpose 'stitch3D'.

    Args:
    masks (np.ndarray): Array of 2D masks.
    master_mask (np.ndarray): 2D master mask used for comparison.
    stitch_threshold (float): Threshold value for stitching.

    Returns:
    np.ndarray: Stitched masks as a 3D volume.
    """
    mmax = masks[0].max()
    empty = 0

    for i in range(len(masks)):
        iou = _intersection_over_union(masks[i], master_mask)[1:, 1:]
        if not iou.size and empty == 0:
            mmax = masks[i].max()
        elif not iou.size and not empty == 0:
            icount = masks[i].max()
            istitch = np.arange(mmax + 1, mmax + icount + 1, 1, int)
            mmax += icount
            istitch = np.append(np.array(0), istitch)
            masks[i] = istitch[masks[i]]
        else:
            iou[iou < stitch_threshold] = 0.0
            iou[iou < iou.max(axis=0)] = 0.0
            istitch = iou.argmax(axis=1) + 1
            ino = np.nonzero(iou.max(axis=1) == 0.0)[0]
            istitch[ino] = 0
            istitch = np.append(np.array(0), istitch)
            masks[i] = istitch[masks[i]]
            empty = 1
    return masks

def dice_coefficient(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Calculate the dice coefficient between two binary masks.

    Args:
    mask1 (np.ndarray): The first binary mask.
    mask2 (np.ndarray): The second binary mask.

    Returns:
    float: The dice coefficient between the two masks.
    """
    intersection = np.logical_and(mask1, mask2)
    return 2 * intersection.sum() / (mask1.sum() + mask2.sum())

def calculate_dice_coef(mask: np.ndarray, shape_thres_percent: float) -> np.ndarray:
    """
    Calculates the Dice coefficient for each mask in the input array and removes masks below the threshold.

    Args:
    mask (np.ndarray): Input array of masks.
    shape_thres_percent (float): Threshold percentage for mask similarity.

    Returns:
    np.ndarray: Array of masks after removing masks below the threshold.
    """
    # Convert to boolean
    mask_val = np.amax(mask)
    mask = mask.astype(bool)
    # Create a median mask as a ref
    med_mask = np.median(mask, axis=0)
    # Check mask similarity
    for i in range(mask.shape[0]):
        # Calculate dc
        dc = dice_coefficient(med_mask, mask[i])
        # Remove mask if below threshold
        if dc < shape_thres_percent:
            mask[i] = 0
    mask = mask.astype('uint16')
    mask[mask != 0] = mask_val
    return mask

def check_mask_similarity(mask: np.ndarray, shape_thres_percent: float = 0.9) -> np.ndarray:
    """
    Check the similarity of a mask by calculating the dice coefficient for each object in the mask.

    Args:
        mask (np.ndarray): The input mask with 3 dimensions.
        shape_thres_percent (float, optional): The threshold percentage for shape similarity. Defaults to 0.9.

    Returns:
        np.ndarray: The new mask with similarity scores over 0.9 for each object.

    Raises:
        TypeError: If the input mask does not contain 3 dimensions.
    """
    print('  ---> Checking mask similarity')
    
    
    # Check that mask contains 3 dimensions
    if mask.ndim != 3:
        raise TypeError(f"Input mask must contain 3 dimensions, {mask.ndim} dim were given")

    # Get the region properties of the mask, and zip it: (label,(slice_f,slice_y,slice_x))
    props: list[tuple[int,tuple]] = list(zip(*regionprops_table(mask,properties=('label','slice')).values()))
    new_mask = np.zeros(shape=mask.shape)
    
    # Process each mask in parallel, and provide a lock
    fixed_args = {'mask':mask, 
                  'shape_thres_percent':shape_thres_percent}
    temp_masks = run_multithread(process_mask,props,fixed_args)
            
    # Reconstruct original size mask
    for slice_obj,temp in temp_masks:
        new_mask[slice_obj] += temp
    return new_mask.astype('uint16')
        
def process_mask(prop: tuple[int,tuple[slice]], mask: np.ndarray, shape_thres_percent: float, lock: Lock) -> tuple[tuple[slice],np.ndarray]:
    # Crop stack to save memory
    obj,slice_obj = prop
    temp = mask[slice_obj].copy()
    # Isolate mask obj
    temp[temp != obj] = 0
    # Calculate dice coef
    with lock:
        temp = calculate_dice_coef(temp, shape_thres_percent)
    return slice_obj,temp

def save_mask(frame: int, mask_stack: np.ndarray, mask_paths: list[PathLike], metadata: dict)-> None:
    # Get the mask path
    path = mask_paths[frame]
    mask_path = path.replace('_Cellpose','_IoU_Track').replace('_Threshold','_IoU_Track')
    # Save the mask
    save_tif(mask_stack[frame].astype('uint16'),mask_path,**metadata)

def unpack_kwargs(kwargs: dict)-> dict:
    """Function to unpack the kwargs and extract necessary variable."""
    if not kwargs:
        return {'finterval':None, 'um_per_pixel':None}
    
    # Unpack the kwargs
    metadata = {}
    for k,v in kwargs.items():
        if k in ['um_per_pixel','finterval']:
            metadata[k] = v
    
    # if kwargs did not contain metadata, set to None
    if not metadata:
        metadata = {'finterval':None, 'um_per_pixel':None}
    
    return metadata


if __name__ == "__main__":
    from tifffile import imread
    from time import time
    
    folder = '/home/Test_images/bigy/HEKA_c1031_c1829_miniSOG_80%_435_2min_40min_002_Merged_s1/Masks_Cellpose'
    mask_folder_src = [join(folder,file) for file in sorted(listdir(folder)) if file.endswith('.tif')]
    mask_stack = load_stack(mask_folder_src,'RFP',range(126),True)
    
    stitch_thres_percent = 0.1
    shape_thres_percent = 0.95
    mask_appear = 5
    copy_first_to_start = True
    copy_last_to_end = True
    
    start = time()
    iou_tracking(mask_folder_src,'YFP',stitch_thres_percent,shape_thres_percent,True,mask_appear,copy_first_to_start,copy_last_to_end)
    
    
    # mask_stack = track_cells(mask_stack,stitch_thres_percent)
    # imwrite('/home/Test_images/masks/tracked_masks.tif',mask_stack.astype('uint16'))
    # # Check shape similarity to avoid false masks
    # mask_stack = check_mask_similarity(mask_stack,shape_thres_percent)
    # imwrite('/home/Test_images/masks/similar_masks.tif',mask_stack.astype('uint16'))
    
    # # Re-assign the new value to the masks and obj. Previous step may have created dicontinuous masks
    # # Morph missing masks
    # # mask_stack = imread('/home/Test_images/masks/labeled_masks.tif')
    # mask_stack = complete_track(mask_stack,mask_appear,copy_first_to_start,copy_last_to_end)
    # imwrite('/home/Test_images/masks/complete_mask.tif', mask_stack.astype('uint16'))
    # print('  ---> Reassigning masks value')
    # mask_stack,_,_ = relabel_sequential(mask_stack)
    # imwrite('/home/Test_images/masks/labeled_masks.tif',mask_stack.astype('uint16'))
    start2 = time()
    print(f"Time to process: {round(start2-start,ndigits=3)} sec\n")
    # mask_stack = check_mask_similarity(mask_stack,shape_thres_percent)
    # imwrite('/home/Test_images/masks/similar_masks2.tif',mask_stack.astype('uint16'))
    # print('  ---> Reassigning masks value')
    # mask_stack,_,_ = relabel_sequential(mask_stack)
    # imwrite('/home/Test_images/masks/labeled_masks2.tif',mask_stack.astype('uint16'))
    # mask_stack = complete_track(mask_stack,mask_appear,copy_first_to_start,copy_last_to_end)
    # imwrite('/home/Test_images/masks/complete_mask2.tif', mask_stack.astype('uint16'))
    # start3 = time()
    # print(f"Time to process: {round(start3-start2,ndigits=3)} sec\n")
    