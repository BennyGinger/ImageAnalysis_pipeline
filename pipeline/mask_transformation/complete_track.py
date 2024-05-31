from __future__ import annotations
from threading import Lock
import numpy as np
from skimage.measure import regionprops_table
from tqdm import tqdm
from pipeline.mask_transformation.mask_warp import mask_warp
from pipeline.image_handeling.data_utility import run_multithread

################## main functions ##################
def complete_track(mask_stack: np.ndarray, mask_appear: int, copy_first_to_start: bool=True, copy_last_to_end: bool=True)-> np.ndarray:
    """Function that completes the tracked masks to fill in the missing masks. 
    Missing masks are filled by morphing the last mask of the track into the new one.

    Args:
        mask_stack (np.ndarray): The input mask stack.
        mask_appear (int): The minimum value of appearance of a mask. If below this threshold, the mask track will be deleted.
        copy_first_to_start (bool, optional): Whether to copy the first mask to the start of the track if it is missing. Defaults to True.
        copy_last_to_end (bool, optional): Whether to copy the last mask to the end of the track if it is missing. Defaults to True.

    Returns:
        np.ndarray: The morphed mask stack.
    """
    
    # Split mask into complete and non-complete tracks
    new_stack = mask_stack.copy()
    incomplete_tracks = trim_incomplete_track(new_stack)
    # If all tracks are complete, return the original mask
    if incomplete_tracks.size == 0:
        return mask_stack
    
    # If not all tracks are complete, remove the complete tracks from the mask stack
    mask_stack -= new_stack
    
    # Get the region properties of the mask, and zip it: (label,(slice_f,slice_y,slice_x))
    props: list[zip[tuple[int,tuple]]] = list(zip(*regionprops_table(mask_stack,properties=('label','slice')).values()))
    
    # Generate input data
    print('  ---> Morphing missing masks')
    fixed_args = {'mask_stack':mask_stack, 
                  'mask_appear':mask_appear, 
                  'copy_first_to_start':copy_first_to_start, 
                  'copy_last_to_end':copy_last_to_end}
    temp_masks = run_multithread(apply_filling,props,fixed_args)
    
    # Reconstruct original size mask
    print('  ---> Reconstructing mask')
    for slice_obj,temp in tqdm(temp_masks):
        # Get the array of the mask that are equals to zero
        new_stack_zerro = new_stack[slice_obj] == 0
        # Add the temp to the new_stack only where the new_stack is zero (leave the rest as is)
        new_stack[slice_obj][new_stack_zerro] += temp[new_stack_zerro]
            
    
    # Trim incomplete tracks, as complete tracks can be overwritten by incomplete tracks
    if copy_first_to_start or copy_last_to_end:
        print('  ---> Trimming incomplete tracks')
        trim_incomplete_track(new_stack)
    return new_stack.astype('uint16')

################## Helper functions ##################
def copy_first_last_mask(mask_stack: np.ndarray, copy_first_to_start: bool=True, copy_last_to_end: bool=True)-> np.ndarray:
    """Function to copy the first and/or last mask to the start and/or end of the stack.
    Args:
        mask_stack (np.ndarray): Mask array.
        copy_first_to_start (bool): Copy the first mask to the start of the stack.
        copy_last_to_end (bool): Copy the last mask to the end of the stack.
    Returns:
        np.ndarray: Mask array with copied masks."""
    # Check no copy needed, skip
    if not copy_first_to_start and not copy_last_to_end:
        return mask_stack
    
    # Convert array to bool, whether mask is present or not
    is_masks = np.any(mask_stack,axis=(1,2))
    
    # if both end of the stack are not missing, return the original stack
    if is_masks[0] and is_masks[-1]:
        return mask_stack
    
    if copy_first_to_start and not is_masks[0]:
        # get the index of the first mask
        idx = np.where(is_masks)[0][0]
        # copy the first mask to the start of the stack
        mask_stack[:idx,...] = mask_stack[idx]
    
    if copy_last_to_end and not is_masks[-1]:
        # get the index of the last mask
        idx = np.where(is_masks)[0][-1]    
        # copy the last mask to the end of the stack
        mask_stack[idx+1:,...] = mask_stack[idx]
    return mask_stack
    
def find_gaps(cropped_stack: np.ndarray)-> list[tuple[int,int,int]]:
    """Function to determine the gaps between masks in a stack.
    Args:
        cropped_stack (np.ndarray): Cropped 3D ndarray containing the mask stack. 

    Returns:
        list[tuple[int,int,int]]: List of tuples containing the last and first mask surrounding the gap, as well as the gap length."""

    # Convert array to bool, whether mask is present or not
    is_masks = np.any(cropped_stack,axis=(1,2))
    
    # If no gap return empty list
    if np.all(is_masks):
        return []
    
    # Find the differences between consecutive elements
    diff = np.diff(is_masks.astype(int))
    
    # Find the indices of the 1s and -1s, and add 1 to them to get the original index
    gap_starts = list(np.where(diff == -1)[0] + 1)
    gap_ends = list(np.where(diff == 1)[0] + 1)
    # If the first mask is missing, remove the first end, as the first starts is missing
    if not is_masks[0]:
        gap_ends.pop(0)
    # Get the gap list
    gap_lst = list(zip(gap_starts, gap_ends))
    return [(start, end, end-start) for start, end in gap_lst]
    
def fill_gaps(cropped_stack: np.ndarray, copy_first_to_start: bool=True, copy_last_to_end: bool=True)-> np.ndarray:
    """This function finds and fills the gaps in the mask stack by morphing the last mask of the track into the new one.

    Args:
        cropped_stack (np.array): Cropped mask array with potential missing frames. 

    Returns:
        stack (np.array): Cropped mask array with filled frames.
    """
    # Copy the first and/or last masks to the ends of the stacks if empty
    cropped_stack = copy_first_last_mask(cropped_stack, copy_first_to_start, copy_last_to_end)
    # Get the indexes of the masks to morph (i.e. that suround empty frames)
    masks_to_morph = find_gaps(cropped_stack)
    # Morph and fill stack
    for id_start,id_end,gap in masks_to_morph:
        n_masks = mask_warp(cropped_stack[id_start-1],cropped_stack[id_end],gap)
        # fill the gaps
        cropped_stack[id_start:id_end,...] = n_masks
    return cropped_stack

def apply_filling(prop: tuple[int,tuple[slice]], mask_stack: np.ndarray, mask_appear: int,copy_first_to_start: bool, copy_last_to_end: bool, lock: Lock)-> tuple[tuple[slice],np.ndarray]:
    """Intermediate function to apply the filling of the gaps in the mask stack in parallel."""
    # Unpack the properties
    obj,slice_obj = prop
    # Modify the crop slice, to include the whole stack. Use lock to avoid shared memory issues
    with lock:
        ref_f = slice(0,mask_stack.shape[0])
        slice_obj=(ref_f, *(slice_obj[1:]))
        temp = mask_stack[slice_obj].copy()
    # Isolate mask obj
    temp[temp!=obj] = 0
    framenumber = len(np.unique(np.where(mask_stack == obj)[0]))
    # If any mask is missing and that the mask appear more than n_mask, fill the gaps
    if framenumber!=mask_stack.shape[0] and framenumber >= mask_appear:
        temp = fill_gaps(temp,copy_first_to_start,copy_last_to_end)
    return slice_obj,temp

def trim_incomplete_track(array: np.ndarray)-> np.ndarray:
    """Function to trim the incomplete tracks from the mask stack.
    Modifies the input array in place.
    
    Args:
        array (np.ndarray): 3D Mask array in tyx format.
    Returns:
        np.ndarray: The list of objects removed."""
    # Make a list of unique objects
    lst_obj = [np.unique(frame) for frame in array]
    lst_obj = np.concatenate(lst_obj) # Flatten the list
    # Count the number of occurences of each object
    obj,cnt = np.unique(lst_obj,return_counts=True)
    # Create a list of obj to remove
    obj_to_remove = obj[cnt!=array.shape[0]]
    array_to_remove = np.isin(array,obj_to_remove)
    array[array_to_remove] = 0
    return obj_to_remove


if __name__ == "__main__":
    from os import listdir
    import sys
    sys.path.append('/home/ImageAnalysis_pipeline/pipeline')
    from image_handeling.data_utility import load_stack
    from os.path import join
    from tifffile import imwrite, imread
    from mask_warp import mask_warp
    from skimage.draw import disk
    import matplotlib.pyplot as plt
    
    mask_stack = imread('/home/Test_images/masks/similar_masks.tif')
    mask_stack = complete_track(mask_stack, 5, True, True)
    imwrite('/home/Test_images/masks/complete_mask.tif', mask_stack.astype('uint16'))