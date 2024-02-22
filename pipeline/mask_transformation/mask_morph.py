from __future__ import annotations
from os import getcwd
import sys
parent_dir = getcwd()
sys.path.append(parent_dir)

import numpy as np
import cv2, itertools
from mahotas import distance
from concurrent.futures import ThreadPoolExecutor


def fill_gaps(mask_stack: np.ndarray)-> np.ndarray:
    """
    This function determine how many missing frames (i.e. empty frames, with no masks) there are from a stack.
    It will then fill the gaps using mask_warp().

    Args:
        stack (np.array): Mask array with missing frames. 

    Returns:
        stack (np.array): Mask array with filled frames.
    """
    # Find the frames with masks and without for a given obj: bool
    is_masks = [np.any(i) for i in mask_stack]
    # Identify masks that suround empty frames
    masks_loc = []
    for i in range(len(is_masks)):
        if (i == 0) or (i == len(is_masks)-1):
            if is_masks[i]==False:
                masks_loc.append(0)
            else:
                masks_loc.append(1)
        else:
            if (is_masks[i]==True) and (is_masks[i-1]==False):
                masks_loc.append(1) # i.e. first mask after empty
            elif (is_masks[i]==True) and (is_masks[i+1]==False):
                masks_loc.append(1) # i.e. last mask before empty
            elif (is_masks[i]==True) and (is_masks[i-1]==True):
                masks_loc.append(2) # already filled with masks
            elif (is_masks[i]==False):
                masks_loc.append(0)

    # Get the key masks
    masks_id = [i for i in range(len(masks_loc)) if masks_loc[i] == 1]

    # Copy the first and/or last masks to the ends of the stacks if empty
    mask_stack[:masks_id[0],...] = mask_stack[masks_id[0],...]
    mask_stack[masks_id[-1]:,...] = mask_stack[masks_id[-1],...]

    # Get the indexes of the masks to morph (i.e. that suround empty frames) and the len of empty gap
    masks_to_morph = []
    for i in range(len(masks_id)-1):
        if any([i in [0] for i in masks_loc[masks_id[i]+1:masks_id[i+1]]]):
            masks_to_morph.append([masks_id[i],masks_id[i+1],len(masks_loc[masks_id[i]+1:masks_id[i+1]])])

    # Morph and fill stack
    for i in masks_to_morph:
        n_masks = mask_warp(mask_stack[i[0]],mask_stack[i[1]],i[2])
        mask_stack[i[0]+1:i[1],...] = n_masks
    return mask_stack

def move_mask_to_center(mask: np.ndarray, midpoint_x: int, midpoint_y: int)-> np.ndarray:
    # Get centroid of mask
    moment_mask = cv2.moments(mask)
    center_x = int(moment_mask["m10"] / moment_mask["m00"])
    center_y = int(moment_mask["m01"] / moment_mask["m00"])

    # Get interval of centroid
    interval_y = midpoint_y-center_y
    interval_x = midpoint_x-center_x

    points_y,points_x = np.where(mask!=0)
    
    # Check that it stays within borders of array
    new_points_y = points_y+interval_y
    new_points_y[new_points_y<0] = 0
    new_points_y[new_points_y>mask.shape[0]-1] = mask.shape[0]-1
    new_points_y = new_points_y.astype(int)
    
    new_points_x = points_x+interval_x
    new_points_x[new_points_x<0] = 0
    new_points_x[new_points_x>mask.shape[1]-1] = mask.shape[1]-1
    new_points_x = new_points_x.astype(int)
    
    # Move the obj
    n_masks = np.zeros((mask.shape))
    obj_val = int(list(np.unique(mask))[1])
    for points in list(zip(new_points_y,new_points_x)):
        n_masks[points] = obj_val
    return n_masks,(center_y,center_x)

def mask_warp(mask_start: np.ndarray, mask_end: np.ndarray, ngap: int)-> np.ndarray:
    # Get middle of array
    midpoint_x = int(mask_start.shape[1]/2)
    midpoint_y = int(mask_start.shape[0]/2)

    new_mask_start,center_coord_mask_start = move_mask_to_center(mask_start,midpoint_x,midpoint_y)
    new_mask_end,center_coord_mask_end = move_mask_to_center(mask_end,midpoint_x,midpoint_y)

    # Centroids linespace
    gap_center_x_coord = np.linspace(center_coord_mask_start[1],center_coord_mask_end[1],ngap+2)
    gap_center_y_coord = np.linspace(center_coord_mask_start[0],center_coord_mask_end[0],ngap+2)

    overlap, crop_slice = bbox_ND(new_mask_start+new_mask_end)

    # Crop and get the overlap of both mask
    new_mask_start_cropped = new_mask_start[crop_slice]
    new_mask_end_cropped = new_mask_end[crop_slice]
    overlap[overlap!=np.max(new_mask_start)+np.max(new_mask_end)] = 0

    # Get the ring (i.e. non-overlap area of each mask)
    ring_start = get_ring_mask(new_mask_start_cropped,overlap)
    ring_end = get_ring_mask(new_mask_end_cropped,overlap)

    if np.any(ring_start!=0) or np.any(ring_end!=0):  #check for different shapes, otherwise just copy shape
        # Get the distance transform of the rings with overlap as 0 (ref point)
        # dt = distance_transform_bf(np.logical_not(overlap))
        dmap_start = get_dmap_array(ring_start,overlap)
        dmap_end = get_dmap_array(ring_end,overlap)

        # Create the increment for each mask, i.e. the number of step needed to fill the gaps
        # if max == 0, then it means that mask is completly incorporated into the other one and will have no gradient
        inc_points_start = get_increment_points(dmap_start,ngap,is_start=True)
        inc_points_end = get_increment_points(dmap_end,ngap,is_start=False)
        
        # Fill the gaps
        masks_list = []
        for i in range(ngap):
            # Select part of the mask that falls out and reset pixel vals to 1        
            overlap_mask_start = create_overlap_mask(overlap,dmap_start,inc_points_start,i)
            overlap_mask_end = create_overlap_mask(overlap,dmap_end,inc_points_end,i)
            
            # Recreate the full shape
            mask = overlap_mask_start+overlap_mask_end
            mask[mask!=0] = np.max(new_mask_start)

            # Resize the mask
            resized_mask = np.zeros((mask_start.shape))
            resized_mask[crop_slice] = mask

            # Replace mask to new center position
            resized_mask,_ = move_mask_to_center(mask=resized_mask,midpoint_x=np.round(gap_center_x_coord[i+1]),midpoint_y=np.round(gap_center_y_coord[i+1]))

            # append the list
            masks_list.append(resized_mask)
    else:
        # Fill the gaps
        masks_list = []
        for i in range(ngap):
            mask = overlap.copy()

            # Resize the mask
            resized_mask = np.zeros((mask_start.shape))
            resized_mask[crop_slice] = mask

            # Replace mask to new center pisotion
            resized_mask,__ = move_mask_to_center(mask=resized_mask,midpoint_x=np.round(gap_center_x_coord[i+1]),midpoint_y=np.round(gap_center_y_coord[i+1]))

            # append the list
            masks_list.append(resized_mask)
    return masks_list

def get_ring_mask(mask: np.ndarray, overlap: np.ndarray)-> np.ndarray:
    ring_mask = mask+overlap
    ring_mask[ring_mask!=np.max(mask)] = 0
    return ring_mask

def get_dmap_array(ring_mask: np.ndarray, overlap: np.ndarray)-> np.ndarray:
    dmap = distance(np.logical_not(overlap),metric='euclidean')
    dmap_array = dmap.copy()
    dmap_array[ring_mask==0] = 0
    return dmap_array

def get_increment_points(dmap_array: np.ndarray, ngap: int, is_start: bool)-> list:
    max_dmap_val = np.max(dmap_array)
    if max_dmap_val == 0:
        return
    if is_start:
        inc_point_list = list(np.linspace(max_dmap_val, 0, ngap+1, endpoint=False))
    else:
        inc_point_list = list(np.linspace(0, max_dmap_val, ngap+1, endpoint=False))
    inc_point_list.pop(0)
    return inc_point_list

def create_overlap_mask(overlap: np.ndarray, dmap_array: np.ndarray, inc_points_list: list | None, gap_index: int)-> np.ndarray:
    max_dmap_val = np.max(dmap_array)
    if max_dmap_val == 0:
        overlap_mask = overlap.copy()
        overlap_mask[overlap_mask!=0] = 1
        return overlap_mask
    
    overlap_mask = dmap_array.copy() 
    overlap_mask[dmap_array > inc_points_list[gap_index]] = 0
    overlap_mask = overlap_mask+overlap
    overlap_mask[overlap_mask!=0] = 1
    return overlap_mask
    
def bbox_ND(mask: np.ndarray)-> tuple[np.ndarray, slice]:
    """
    This function take a np.array (any dimension) and create a bounding box around the nonzero shape.
    Also return a slice object to be able to reconstruct to the originnal shape.

    Args:
        array (np.array): Array containing a single mask. The array can be of any dimension.

    Returns:
        (tuple): Tuple containing the new bounding box array and the slice object used for the bounding box. 
    """
    # Determine the number of dimensions
    N = mask.ndim
    
    # Go trhough all the axes to get min and max coord val
    slice_list = []
    for ax in itertools.combinations(reversed(range(N)), N - 1):
        nonzero = np.any(mask, axis=ax)
        vmin, vmax = np.where(nonzero)[0][[0, -1]]
        # Store these coord as slice obj
        slice_list.append(slice(vmin,vmax+1))
    
    s = tuple(slice_list)
    
    return tuple([mask[s], s])

def apply_morph(input_data: list)-> np.ndarray:
    mask_stack,obj,n_mask = input_data
    temp = mask_stack.copy()
    temp[temp!=obj] = 0
    framenumber = len(np.unique(np.where(mask_stack == obj)[0]))
    if framenumber!=mask_stack.shape[0] and framenumber > n_mask:
        temp = fill_gaps(temp)
    return temp

# # # # # # # # main functions # # # # # # # # # 
def morph_missing_mask(mask_stack: np.ndarray, n_mask: int)-> np.ndarray:
    print('  ---> Morphing missing masks')
    input_data = [(mask_stack,obj,n_mask) for obj in list(np.unique(mask_stack))[1:]]
    
    with ThreadPoolExecutor() as executor:
        temp_masks = executor.map(apply_morph,input_data)
        new_stack = np.zeros((mask_stack.shape))
        for obj,temp in zip(list(np.unique(mask_stack))[1:],temp_masks):
            new_stack = new_stack + temp
            if np.any(new_stack>obj):
                new_stack[new_stack>obj] = new_stack[new_stack>obj]-obj
    #FIXME this is equal to the trim_mask function from iou_tracking. Would be good to have the trim in a seperate function. Not always run it, when doing the morph
    # Recheck for incomplete track   
    for obj in list(np.unique(new_stack))[1:]:
        framenumber = len(np.unique(np.where(new_stack==obj)[0]))
        if framenumber!=mask_stack.shape[0]:
            new_stack[new_stack==obj] = 0
    return new_stack.astype('uint16')

# def morph_missing_mask(mask_stack: np.ndarray, n_mask: int)-> np.ndarray:
#     print('  ---> Morphing missing masks')
#     new_stack = np.zeros((mask_stack.shape))
#     for obj in list(np.unique(mask_stack))[1:]:
#         temp = mask_stack.copy()
#         temp[temp!=obj] = 0
#         framenumber = len(np.unique(np.where(mask_stack == obj)[0]))
#         if framenumber!=mask_stack.shape[0] and framenumber > n_mask:
#             temp = fill_gaps(temp)
#         new_stack = new_stack + temp
#         if np.any(new_stack>obj):
#             new_stack[new_stack>obj] = new_stack[new_stack>obj]-obj
    
#     # Recheck for incomplete track
#     for obj in list(np.unique(new_stack))[1:]:
#         framenumber = len(np.unique(np.where(new_stack==obj)[0]))
#         if framenumber!=mask_stack.shape[0]:
#             new_stack[new_stack==obj] = 0
    
#     return new_stack.astype('uint16') 