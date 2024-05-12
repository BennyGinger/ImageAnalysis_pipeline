from __future__ import annotations
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from skimage.measure import regionprops_table
from pipeline.mask_transformation.mask_warp import mask_warp
from functools import partial

################## main functions ##################
def complete_track(mask_stack: np.ndarray, mask_appear: int, copy_first_to_start: bool=True, copy_last_to_end: bool=True) -> np.ndarray:
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
    
    # Get the region properties of the mask, and zip it: (label,(slice_f,slice_y,slice_x))
    props: zip[tuple[int,tuple]] = zip(*regionprops_table(mask_stack,properties=('label','slice')).values())
    
    # Generate input data
    print('  ---> Morphing missing masks')
    apply_filling_partial = partial(apply_filling, mask_stack=mask_stack, mask_appear=mask_appear, 
                                    copy_first_to_start=copy_first_to_start, copy_last_to_end=copy_last_to_end)
    
    # Apply morphing
    with ThreadPoolExecutor() as executor:
        temp_masks = executor.map(apply_filling_partial, props)
    
    # Reconstruct original size mask
    new_stack = np.zeros((mask_stack.shape))
    for obj,slice_obj,temp in temp_masks:
        new_stack[slice_obj] += temp
        # Trim any overlapping mask
        if np.any(new_stack > obj):
            new_stack[new_stack > obj] = new_stack[new_stack > obj] - obj
    
    # Trim incomplete tracks, as complete tracks can be overwritten by incomplete tracks
    if copy_first_to_start or copy_last_to_end:
        print('  ---> Trimming incomplete tracks')
        new_stack = trim_incomplete_track(new_stack)
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
    # Check if any copy is needed
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

def apply_filling(prop: tuple[int,tuple[slice]], mask_stack: np.ndarray, mask_appear: int,
                  copy_first_to_start: bool, copy_last_to_end: bool)-> tuple[int,tuple[slice],np.ndarray]:
    """Intermediate function to apply the filling of the gaps in the mask stack in parallel."""
    # Crop stack to save memory
    obj,slice_obj = prop
    temp = mask_stack[slice_obj].copy()
    # Isolate mask obj
    temp[temp!=obj] = 0
    framenumber = len(np.unique(np.where(mask_stack == obj)[0]))
    # If any mask is missing and that the mask appear more than n_mask, fill the gaps
    if framenumber!=mask_stack.shape[0] and framenumber > mask_appear:
        temp = fill_gaps(temp,copy_first_to_start,copy_last_to_end)
    return obj,slice_obj,temp

def trim_incomplete_track(array: np.ndarray)-> np.ndarray:
    """Function to trim the incomplete tracks from the mask stack.
    Args:
        array (np.ndarray): 3D Mask array in tyx format.
    Returns:
        np.ndarray: Mask array with incomplete tracks removed."""
    # Make a list of unique objects
    lst_obj = [np.unique(frame) for frame in array]
    lst_obj = np.concatenate(lst_obj) # Flatten the list
    # Count the number of occurences of each object
    obj,cnt = np.unique(lst_obj,return_counts=True)
    # Create a list of obj to remove
    obj_to_remove = obj[cnt!=array.shape[0]]
    for obj in obj_to_remove:
        array[array==obj] = 0
    return array

if __name__ == "__main__":
    from os import listdir
    import sys
    sys.path.append('/home/ImageAnalysis_pipeline/pipeline')
    from image_handeling.data_utility import load_stack
    from os.path import join
    from tifffile import imwrite, imread
    from mask_warp import mask_warp
    
    # mask_folder = '/home/Test_images/nd2/Run2/c2z25t23v1_nd2_s1/Masks_IoU_Track'
    # mask_lst = [join(mask_folder,img) for img in sorted(listdir(mask_folder))]
    # channel_list = ['RFP']
    # frame_range = range(23)
    
    # mask = load_stack(mask_lst,channel_list,frame_range)
    # print(mask.shape)
    # mask[mask!=14] = 0
    # mask[:3, :, :] = 0
    # mask[4:5, :, :] = 0
    
    # imwrite('/home/Test_images/masks/input.tif', mask.astype('uint16'))
    
    # new_mask = morph_missing_mask(mask,5,True,True)
    # imwrite('/home/Test_images/masks/output.tif', new_mask.astype('uint16'))
    
    # from skimage.draw import disk
    # def mask_stack():
    #     m1 = np.zeros((3, 100, 100), dtype=np.uint8)
    #     rr, cc = disk((40, 50), 14)
    #     m1[:,rr, cc] = 1
    #     m2 = np.zeros((3, 100, 100), dtype=np.uint8)
    #     rr, cc = disk((50, 50), 20)
    #     m2[:,rr, cc] = 1
    #     m3 = np.zeros((4, 100, 100), dtype=np.uint8)
    #     rr, cc = disk((50, 60), 15)
    #     m3[:,rr, cc] = 1
    #     mask = np.concatenate((m1,m2,m3),axis=0)
    #     return mask
    
    # mask = mask_stack()
    # imwrite('/home/Test_images/masks/expected.tif', mask.astype('uint16'))
    
    # mask[2:4, :, :] = 0
    # imwrite('/home/Test_images/masks/input.tif', mask.astype('uint16'))
    
    # new_mask = complete_track(mask,True,True)
    # imwrite('/home/Test_images/masks/output.tif', new_mask.astype('uint16'))
    mask_appear = 5
    copy_first_to_start = True
    copy_last_to_end = True
    mask_stack = imread('/home/Test_images/masks/labeled_masks.tif')
    complete_track(mask_stack, mask_appear, copy_first_to_start, copy_last_to_end)