from __future__ import annotations
import numpy as np
import cv2, itertools
from mahotas import distance
from concurrent.futures import ThreadPoolExecutor


def copy_first_last_mask(mask_stack: np.ndarray, copy_first_to_start: bool=True, copy_last_to_end: bool=True)-> np.ndarray:
    """Function to copy the first and/or last mask to the start and/or end of the stack.
    Attributes:
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
    
def get_masks_to_morph_lst(mask_stack: np.ndarray)-> list[tuple[int,int,int]]:
    """Function to determine the gaps between masks in a stack.
    Args:
        mask_stack (np.ndarray): 3D ndarray containing the mask stack.

    Returns:
        list[tuple[int,int,int]]: List of tuples containing the last and first mask surrounding the gap, as well as the gap length."""

    # Convert array to bool, whether mask is present or not
    is_masks = np.any(mask_stack,axis=(1,2))
    
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
    
def fill_gaps(mask_stack: np.ndarray, copy_first_to_start: bool=True, copy_last_to_end: bool=True)-> np.ndarray:
    """
    This function determine how many missing frames (i.e. empty frames, with no masks) there are from a stack.
    It will then fill the gaps using mask_warp().

    Args:
        stack (np.array): Mask array with missing frames. 

    Returns:
        stack (np.array): Mask array with filled frames.
    """
    # Copy the first and/or last masks to the ends of the stacks if empty
    mask_stack = copy_first_last_mask(mask_stack, copy_first_to_start, copy_last_to_end)
    # Get the indexes of the masks to morph (i.e. that suround empty frames)
    masks_to_morph = get_masks_to_morph_lst(mask_stack)
    # Morph and fill stack
    for id_start,id_end,gap in masks_to_morph:
        n_masks = mask_warp(mask_stack[id_start-1],mask_stack[id_end],gap)
        # fill the gaps
        mask_stack[id_start:id_end,...] = n_masks
    return mask_stack

def move_mask_to_center(mask: np.ndarray, midpoint_x: int, midpoint_y: int)-> tuple[np.ndarray,tuple[int,int]]:
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

def mask_warp(mask_start: np.ndarray, mask_end: np.ndarray, ngap: int)-> list[np.ndarray]:
    
    # Get middle of array
    midpoint_x = int(mask_start.shape[1]/2)
    midpoint_y = int(mask_start.shape[0]/2)

    mask_start_centered,center_coord_mask_start = move_mask_to_center(mask_start,midpoint_x,midpoint_y)
    mask_end_centered,center_coord_mask_end = move_mask_to_center(mask_end,midpoint_x,midpoint_y)
    
    # Centroids linespace
    gap_center_x_coord = np.linspace(center_coord_mask_start[1],center_coord_mask_end[1],ngap+2).astype(int)
    gap_center_y_coord = np.linspace(center_coord_mask_start[0],center_coord_mask_end[0],ngap+2).astype(int)

    overlap, crop_slice = bbox_ND(mask_start_centered+mask_end_centered)
    
    # Crop and get the overlap of both mask
    mask_start_cropped = mask_start_centered[crop_slice]
    mask_end_cropped = mask_end_centered[crop_slice]
    overlap[overlap!=np.amax(mask_start_centered)+np.amax(mask_end_centered)] = 0
    
    # Get the ring (i.e. non-overlap area of each mask)
    ring_start = get_ring_mask(mask_start_cropped,overlap)
    ring_end = get_ring_mask(mask_end_cropped,overlap)
    
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
            mask[mask!=0] = np.amax(mask_start_centered)

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
    mask_stack,obj,n_mask,copyfirst,copylast = input_data
    temp = mask_stack.copy()
    temp[temp!=obj] = 0
    framenumber = len(np.unique(np.where(mask_stack == obj)[0]))
    # If any mask is missing and that the mask appear more than n_mask, fill the gaps
    if framenumber!=mask_stack.shape[0] and framenumber > n_mask:
        temp = fill_gaps(temp,copyfirst,copylast)
    return temp

# # # # # # # # main functions # # # # # # # # # 
def morph_missing_mask(mask_stack: np.ndarray, mask_appear: int, copy_first_to_start: bool=True, copy_last_to_end: bool=True) -> np.ndarray:
    """
    Fills all missing masks in time sequence if the original mask appears more than a certain threshold.
    Missing masks added will be the results of the first appearance morphed into the last appearance.

    Args:
        mask_stack (np.ndarray): The input mask stack.
        mask_appear (int): The minimum value of appearance of a mask. If below this threshold, the mask track will be deleted.
        keep_incomplete_track (bool, optional): Whether to keep incomplete tracks. Defaults to False.

    Returns:
        np.ndarray: The morphed mask stack."""
    
    # Generate input data
    print('  ---> Morphing missing masks')
    input_data = [(mask_stack, 
                   obj, 
                   mask_appear,
                   copy_first_to_start,
                   copy_last_to_end) 
                  for obj in list(np.unique(mask_stack))[1:]]
    # Apply morphing
    with ThreadPoolExecutor() as executor:
        temp_masks = executor.map(apply_morph, input_data)
        new_stack = np.zeros((mask_stack.shape))
        # Combine the morphed masks
        for obj, temp in zip(list(np.unique(mask_stack))[1:], temp_masks):
            new_stack = new_stack + temp
            # Trim any overlapping mask
            if np.any(new_stack > obj):
                new_stack[new_stack > obj] = new_stack[new_stack > obj] - obj

    # Keep incomplete track
    if not copy_first_to_start or not copy_last_to_end:
        return new_stack.astype('uint16')
    
    # Trim incomplete track
    for obj in list(np.unique(new_stack))[1:]:
        framenumber = len(np.unique(np.where(new_stack == obj)[0]))
        if framenumber != mask_stack.shape[0]:
            new_stack[new_stack == obj] = 0

    return new_stack.astype('uint16')

if __name__ == "__main__":
    from os import listdir
    import sys
    sys.path.append('/home/ImageAnalysis_pipeline/pipeline')
    from image_handeling.data_utility import load_stack
    from os.path import join
    from tifffile import imwrite
    
    # mask_folder = '/home/Test_images/nd2/Run2/c2z25t23v1_nd2_s1/Masks_IoU_Track'
    # mask_lst = [join(mask_folder,img) for img in sorted(listdir(mask_folder))]
    # channel_list = ['RFP']
    # frame_range = range(23)
    
    # mask = load_stack(mask_lst,channel_list,frame_range)
    # mask[mask!=14] = 0
    # mask[:3, :, :] = 0
    # mask[4:5, :, :] = 0
    
    # imwrite('/home/Test_images/masks/input.tif', mask.astype('uint16'))
    
    # new_mask = morph_missing_mask(mask,5,True,True)
    # imwrite('/home/Test_images/masks/output.tif', new_mask.astype('uint16'))
    
    from skimage.draw import disk
    def mask_stack():
        img = np.zeros((10, 10, 10), dtype=np.uint8)
        rr, cc = disk((5, 5), 4)
        img[:,rr, cc] = 1
        return img
    
    mask = mask_stack()
    mask[2:4, :, :] = 0
    imwrite('/home/Test_images/masks/input.tif', mask.astype('uint16'))
    
    new_mask = fill_gaps(mask,True,True)
    imwrite('/home/Test_images/masks/output.tif', new_mask.astype('uint16'))