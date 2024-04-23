from __future__ import annotations
import itertools
import cv2
import numpy as np
from mahotas import distance

# TODO: maybe change this into a class...
############################# Main Functions #############################
def mask_warp(mask_start: np.ndarray, mask_end: np.ndarray, ngap: int) -> np.ndarray:
    """Warps a mask from the starting position to the ending position. Represent the missing mask in a track.

    Args:
        mask_start (np.ndarray): The starting mask.
        mask_end (np.ndarray): The ending mask.
        ngap (int): The number of steps in the warp trajectory.

    Returns:
        np.ndarray: The warped mask.

    """
    # Get middle of image, and the original image size
    img_size = mask_start.shape
    img_center = (int(mask_start.shape[0] / 2), int(mask_start.shape[1] / 2))

    # Move the mask to the center
    mask_start, mask_start_centroid = relocate_mask(mask_start, img_center)
    mask_end, mask_end_centroid = relocate_mask(mask_end, img_center)

    # Centroids linespace, to get the trajectory of the mask
    interpol_centroids = interpolate_centroids(mask_start_centroid, mask_end_centroid, ngap)

    # Get the overlap mask, as well as the slice to crop the mask
    overlap, crop_slice = get_overlap_mask(mask_start, mask_end)

    # Crop both mask, to match the overlap image size
    mask_start = mask_start[crop_slice]
    mask_end = mask_end[crop_slice]

    # Get the non-overlap masks
    non_overlap_start, non_overlap_end = get_non_overlap_mask(mask_start, mask_end, overlap)

    # Get the missing masks
    missing_masks = get_missing_masks(overlap, non_overlap_start, non_overlap_end, ngap, np.amax(mask_start))

    # Resize the mask to original size
    return resize_mask(missing_masks, img_size, crop_slice, ngap, interpol_centroids)

############################# Helper Functions #############################
def bbox_ND(mask: np.ndarray)-> tuple[np.ndarray, slice]:
    """
    This function take a np.array (any dimension) and create a bounding box around the nonzero shape.
    Also return a slice object to be able to reconstruct to the original size.

    Args:
        mask (np.ndarray): Array containing a single mask. The array can be of any dimension.

    Returns:
        (tuple[np.ndarray, slice]): Tuple containing the new bounding box array and the slice object used for the bounding box. 
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

def relocate_mask(mask: np.ndarray, location: tuple[int,int])-> tuple[np.ndarray,tuple[int,int]]:
    """Relocates a mask by shifting its position based on the given location.

    Args:
        mask (np.ndarray): The input mask array.
        location (tuple[int,int]): The new location (centroids) to which the mask should be relocated.

    Returns:
        tuple[np.ndarray,tuple[int,int]]: A tuple containing the relocated mask array and the original centroid location.

    """
    loc_y,loc_x = location
    # Get centroid of mask
    moment_mask = cv2.moments(mask)
    center_x = int(moment_mask["m10"] / moment_mask["m00"])
    center_y = int(moment_mask["m01"] / moment_mask["m00"])

    # Get interval of centroid
    diff_y = int(loc_y-center_y)
    diff_x = int(loc_x-center_x)

    points_y,points_x = np.where(mask!=0)
    
    # Check that it stays within borders of array
    #TODO: I'm not sure why we have shape -1 here
    new_points_y = relocate_points(points_y,diff_y,mask.shape[0])
    new_points_x = relocate_points(points_x,diff_x,mask.shape[1])
    
    # Move the obj
    n_masks = np.zeros((mask.shape))
    obj_val = int(np.max(mask))
    for points in list(zip(new_points_y,new_points_x)):
        n_masks[points] = obj_val
    return n_masks,(center_y,center_x)

def relocate_points(points: np.ndarray, shift: int, max_shift: int) -> np.ndarray:
    """Relocates the given points by adding a shift value and constraining them within the range of 0 to max_shift.

    Args:
        points (np.ndarray): The array of points to be relocated.
        shift (int): The shift value to be added to the points.
        max_shift (int): The maximum allowed shift value, i.e. the border of the image.

    Returns:
        np.ndarray: The relocated points array.
    """
    points = points + shift
    # Trim the points if they are out of bounds, i.e. not within the image size
    points[points < 0] = 0
    points[points > max_shift] = max_shift
    return points

def interpolate_centroids(centroid_start: tuple[int,int], centroid_end: tuple[int,int], ngap: int) -> list[tuple[int,int]]:
    """Takes two sets of centroids (start and end) and a given number of gaps.
    Returns a list of new centroids, evenly distributed between the start and end centroids.

    Args:
        centroid_start (tuple[int,int]): Centroid of the start mask.
        centroid_end (tuple[int,int]): Centroid of the end mask.
        ngap (int): The number of gaps between the start and end masks.

    Returns:
        list[tuple[int,int]]: A list of new centroids, evenly distributed between the start and end centroids.
    """
    num = ngap+2
    # y_centroids
    start_y = centroid_start[0]
    stop_y = centroid_end[0]
    y_cent = np.linspace(start_y,stop_y,num)
    # x_centroids
    start_x = centroid_start[1]
    stop_x = centroid_end[1]
    x_cent = np.linspace(start_x,stop_x,num)
    # Zip all
    return list(zip(y_cent,x_cent))

def get_overlap_mask(mask_start: np.ndarray, mask_end: np.ndarray) -> tuple[np.ndarray, slice]:
    """Calculates the overlap mask and crop slice between two input masks.

    Args:
        mask_start (np.ndarray): The starting mask.
        mask_end (np.ndarray): The ending mask.

    Returns:
        tuple[np.ndarray, slice]: A tuple containing the overlap mask and crop slice.
    """
    overlap_mask, crop_slice = bbox_ND(mask_start + mask_end)
    # Remove non-overlapping area
    overlap_val = np.amax(mask_start) + np.amax(mask_end)
    overlap_mask[overlap_mask != overlap_val] = 0
    return overlap_mask, crop_slice

def get_non_overlap_mask(mask_start: np.ndarray, mask_end: np.ndarray, overlap: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Generate the non-overlapping mask for both start and end mask.

    Args:
        mask_start (np.ndarray): The start mask.
        mask_end (np.ndarray): The end mask.
        overlap (np.ndarray): The overlap mask.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the non-overlapping mask for the start and end masks.
    """
    no_mask_start = mask_start + overlap
    no_mask_start[no_mask_start != np.amax(mask_start)] = 0
    no_mask_end = mask_end + overlap
    no_mask_end[no_mask_end != np.amax(mask_end)] = 0
    return no_mask_start, no_mask_end

def get_missing_masks(overlap: np.ndarray, non_overlap_start: np.ndarray, non_overlap_end: np.ndarray, ngap: int, mask_value: int) -> np.ndarray:
    """Generate missing masks between non-overlapping start and end masks.

    Args:
        overlap (np.ndarray): The overlapping mask.
        non_overlap_start (np.ndarray): The non-overlapping start mask.
        non_overlap_end (np.ndarray): The non-overlapping end mask.
        ngap (int): The number of missing masks to generate.
        mask_value (int): The value to assign to the missing masks.

    Returns:
        np.ndarray: The generated missing masks.

    """
    
    # If no difference between the start and end mask, then just copy the overlap
    if np.all(non_overlap_start==0) and np.all(non_overlap_end==0):
        missing_masks = np.stack([overlap]*ngap,axis=0)
        missing_masks[missing_masks!=0] = mask_value
        return missing_masks
    
    # Else build the missing masks
    dmap_start = get_dmap_array(non_overlap_start,overlap)
    dmap_end = get_dmap_array(non_overlap_end,overlap)
    
    # Create the increment for each mask, i.e. the number of step needed to fill the gaps
    inc_points_start = get_increment_points(dmap_start,ngap,is_start=True)
    inc_points_end = get_increment_points(dmap_end,ngap,is_start=False)
    # Create the missing masks
    masks_list = []
    for i in range(ngap):
        # Select part of the mask that falls out and reset pixel vals to 1        
        mask_start = create_mask(overlap,dmap_start,inc_points_start,i)
        mask_end = create_mask(overlap,dmap_end,inc_points_end,i)
        
        # Recreate the full shape
        mask = mask_start+mask_end
        mask[mask!=0] = mask_value
        masks_list.append(mask)
    return np.stack(masks_list,axis=0)

def resize_mask(missing_masks: np.ndarray, img_size: tuple[int,int], crop_slice: slice, ngap: int, interpol_centroids: list[tuple[int,int]])-> np.ndarray:
    """
    Resize the mask to the original size.

    Args:
        missing_masks (np.ndarray): The array of missing masks.
        img_size (tuple[int,int]): The size of the original image.
        crop_slice (slice): The slice used to crop the mask.
        ngap (int): The number of gaps.
        interpol_centroids (list[tuple[int,int]]): The list of interpolated centroids for each missing mask.

    Returns:
        np.ndarray: The resized mask.

    """
    # Resize the mask to original size
    resized_mask = np.zeros((ngap,*img_size))
    for i in range(ngap):
        # Resize the mask
        temp_mask = resized_mask[i]
        temp_mask[crop_slice] = missing_masks[i]
        # Replace mask to new center position, and reset mask value to original
        temp_mask,__ = relocate_mask(temp_mask,interpol_centroids[i+1])
        temp_mask[temp_mask!=0] = np.amax(missing_masks)
        
        # Reinsert in the stack
        resized_mask[i] = temp_mask
    return resized_mask

def get_dmap_array(non_overlap_mask: np.ndarray, overlap: np.ndarray)-> np.ndarray:
    """Get the distance transform of the non_overlap mask with the overlap as reference point (overlap == 0)
    
    Args:
        non_overlap_mask (np.ndarray): The ring mask.
        overlap (np.ndarray): The overlap mask.
    Returns:
        np.ndarray: The distance map array."""
    dmap = distance(np.logical_not(overlap),metric='euclidean')
    dmap_array = dmap.copy()
    dmap_array[non_overlap_mask==0] = 0
    return dmap_array

def get_increment_points(dmap_array: np.ndarray, ngap: int, is_start: bool) -> list[float]:
    """
    Generate a list of increment points based on the given depth map array.

    Args:
        dmap_array (np.ndarray): The depth map array.
        ngap (int): The number of gaps between the increment points.
        is_start (bool): Indicates whether the increment points start from the maximum value or not.

    Returns:
        list: A list of increment points.

    """
    # if max == 0, then it means that mask is completely incorporated into the other one and will have no gradient
    max_dmap_val = np.max(dmap_array)
    if max_dmap_val == 0:
        return
    if is_start:
        inc_point_list = list(np.linspace(max_dmap_val, 0, ngap+1, endpoint=False))
    else:
        inc_point_list = list(np.linspace(0, max_dmap_val, ngap+1, endpoint=False))
    inc_point_list.pop(0)
    return inc_point_list

def create_mask(overlap: np.ndarray, dmap_array: np.ndarray, inc_points_list: list | None, gap_index: int) -> np.ndarray:
    """
    Create the missing mask based on the overlap, depth map array, inclusion points list, and gap index.

    Args:
        overlap (np.ndarray): The overlap array.
        dmap_array (np.ndarray): The depth map array.
        inc_points_list (list | None): The inclusion points list.
        gap_index (int): The gap index.

    Returns:
        np.ndarray: The created mask.

    """
    max_dmap_val = np.max(dmap_array)
    if max_dmap_val == 0:
        overlap_mask = overlap.copy()
        overlap_mask[overlap_mask != 0] = 1
        return overlap_mask

    overlap_mask = dmap_array.copy()
    overlap_mask[dmap_array > inc_points_list[gap_index]] = 0
    overlap_mask = overlap_mask + overlap
    overlap_mask[overlap_mask != 0] = 1
    return overlap_mask