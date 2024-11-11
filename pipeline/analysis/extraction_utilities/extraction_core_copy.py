from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from skimage.measure import regionprops_table
from scipy.spatial import distance
from pipeline.utilities.data_utility import load_stack, get_exp_props
from pipeline.mask_transformation.utils import erode_masks
from typing import TypeVar

# Custom variable type
T = TypeVar('T')


# Build-in properties for regionprops_table. NOTE: if modifying this list, make sure to update the _rename_columns() function.
PROPERTIES = ['area','centroid','intensity_mean',
                       'label','perimeter','slice','solidity']


############### Main function ####################
def extract_regionprops(frame_idx: int, mask_data: dict[str, list[Path]], img_paths: list[Path], do_diff: bool, ref_data: dict[str, list[Path]] | None, class_data: dict[str, list[Path]] | None, diff_channel_ratio: str | None = None, ref_resolution: float | None = None)-> pd.DataFrame:
        """Core function to extract the regionprops from the mask_array and img_array. It will post-process the data if needed.
        
        Args:
            frame_idx (int | None): index of the frame to process.
            
            mask_paths (list[Path]): list of paths to the mask files
            
            mask_name (str): name of the mask
            
            img_paths (list[Path]): list of paths to the image files
            
            do_diff (bool): whether to extract the differencial data in the regionprops
            
            ref_paths_and_resolution (tuple[dict[str, list[Path]], float | None] | None): Tuple containing the reference masks paths stored in a dict and the pixel resolution. Set to None if no reference masks are provided.
            
            class_paths (dict[str, list[Path]] | None): Dictionary containing the classification masks paths stored in a dict. Set to None if no classification masks are provided.
            
            ratio (str | None): ratio to compute the difference between two channels. The format should be 'channel1/channel2' Set to None if no ratio is provided. Only apply if do_diff is True. Default is None.
            
            Returns:
                pd.DataFrame: extracted data from the images."""
        
        # Unpack the mask data
        mask_name, mask_paths = list(mask_data.items())[0]
            
        # Load the image and mask arrays
        prop = base_regionprops(frame_idx, img_paths, mask_paths, do_diff, diff_channel_ratio)
        
        if ref_data:
            ref_props(ref_data, ref_resolution, mask_paths, frame_idx, prop)
        
        if class_data:
            class_props(class_data, mask_paths, frame_idx, prop)
        
        # Return the data as a dataframe       
        df = pd.DataFrame(prop)
        df['frame'] = frame_idx+1
        df['mask_name'] = mask_name
        return df

def base_regionprops(frame_idx: int, img_paths: list[Path], mask_paths: list[Path], do_diff: bool, diff_channel_ratio: str | None = None)-> dict[str,any]:
    # Get img properties
    channels, _, nframes, _ = get_exp_props(img_paths)
    nchannels = len(channels)
    
    # Extract raw regionprops
    if not do_diff or nframes == 1:
        img_array = load_stack(img_paths, frame_range=frame_idx, return_2D=True)
        mask_array = load_stack(mask_paths, frame_range=frame_idx, return_2D=True)
        
    # Extract the differencial regionprops
    else:
        
        if diff_channel_ratio:
            # Validate the ratio
            _validate_channel_ratio(channels, diff_channel_ratio)
            nchannels = 1
            channels = [diff_channel_ratio]
        
        # Load the mask and image arrays
        mask_array, img_array = _load_diff_arrays(img_paths, mask_paths, frame_idx, diff_channel_ratio, nchannels)
    
    # Extract the base regionprops
    prop = regionprops_table(mask_array, img_array, properties=PROPERTIES, separator='_')

    # Rename the columns
    if nchannels > 1:
        col_rename = {f'intensity_mean_{i}': f'diff_intensity_mean_{chan}' for i, chan in enumerate(channels)}
    else:
        col_rename = {f'intensity_mean': f'diff_intensity_mean_{channels[0]}'}
    
    # Rename the props
    prop = {col_rename[key]: value for key, value in prop.items()}

################# Processing functions ####################
def diff_props(img_paths: list[Path], mask_paths: list[Path], frame_idx: int | None, diff_channel_ratio: str | None, prop: dict[str,float])-> None:
    """Function that will substract each frames with the previous frame to extract the difference in the regionprops."""
    
    
    # Get img properties
    channels, _, nframes, _ = get_exp_props(img_paths)
    nchannels = len(channels)
    if diff_channel_ratio:
        # Validate the ratio
        _validate_channel_ratio(channels, diff_channel_ratio)
        nchannels = 1
        channels = [diff_channel_ratio]
    # Return if only one frame
    if nframes == 1:
        return
    
    # Load the mask and image arrays
    mask_array, diff_array = _load_diff_arrays(img_paths, mask_paths, frame_idx, diff_channel_ratio, nchannels)
    
    # Rename the columns
    if nchannels > 1:
        col_rename = {f'intensity_mean_{i}': f'diff_intensity_mean_{chan}' for i, chan in enumerate(channels)}
    else:
        col_rename = {f'intensity_mean': f'diff_intensity_mean_{channels[0]}'}
    
    # Extract the regionprops
    prop_diff = regionprops_table(mask_array,diff_array,properties=['intensity_mean'],separator='_')
    # Rename the props
    prop_diff = {col_rename[key]: value for key, value in prop_diff.items()}
    
    # Update the main properties with the difference
    prop.update(prop_diff)

def ref_props(ref_data: dict[str, list[Path]], resolution: float | None, mask_paths: list[Path], frame_idx: int, prop: dict[str,float])-> None:
    """Extract the regionprops from the reference masks. The function will compute the distance transform value from the dmap mask of the centroid of the primary mask. The distance transform value will be added to the main properties."""
    
    # Load the mask array
    mask_array = load_stack(mask_paths, frame_range=frame_idx, return_2D=True)
    
    # Extract the reference masks
    for ref_name, ref_paths in ref_data.items():
        # Apply the distance transform to the reference array
        ref_array = load_stack(ref_paths, frame_range=frame_idx, return_2D=True)
        
        # Update the main properties with the dmap
        if resolution:
            prop[f'dmap_um_{ref_name}'] = list(_get_min_distance(mask_array,ref_array)*resolution)
        else:
            prop[f'dmap_pixel_{ref_name}'] = list(_get_min_distance(mask_array,ref_array))

def class_props(class_data: dict[str, list[Path]], mask_paths: list[Path], frame_idx: int | None, prop: dict[str, float])-> None:
    """Extract the regionprops from the secondary masks. The function will compute the overlap between the primary mask cells and the secondary masks cells and return a boolean value, whether the primary mask cells are in the secondary masks cells."""
    
    # Load the mask array
    mask_array = load_stack(mask_paths, frame_range=frame_idx, return_2D=True)
    
    for class_name, class_paths in class_data.items():
        # Load the classification array
        class_array = load_stack(class_paths, channels=class_name, frame_range=frame_idx, return_2D=True)
        # Erode the classification masks
        class_array = erode_masks(class_array)
        
        # Extract the regionprops
        prop_sec = regionprops_table(mask_array, class_array, properties=['intensity_max'], separator='_', extra_properties=[label_in])
            
        # Update the main properties with the overlap
        prop[f'{class_name}_classification'] = [f"{class_name}_{label} overlaps" if state else f"no overlap" for state, label in zip(prop_sec['label_in'], prop_sec['intensity_max'])]


################# Helper functions ####################
def _get_min_distance(mask: np.ndarray, ref_array: np.ndarray)-> np.ndarray[float]:
    # Get the stacked coordinates of the mask and the reference array
    mask_coords = np.column_stack(np.where(mask != 0))
    ref_coords = np.column_stack(np.where(ref_array != 0))
    
    # Get the uniques labels
    objects_ids = np.unique(mask)[1:]
    
    # Get the centroids of the masks
    mask_centroids = np.array([np.mean(mask_coords[mask[mask_coords[:,0],mask_coords[:,1]] == obj], axis=0) for obj in objects_ids])
    
    # Compute the minimum distance between each mask and the reference array
    return np.min(distance.cdist(mask_centroids, ref_coords), axis=1)

def _validate_channel_ratio(channels: list[str], ratio: str)-> None:
    ratio_channels = ratio.split('/')
    
    if len(ratio_channels) != 2:
        raise ValueError("The ratio should be in the form 'channel1/channel2'")
    
    for channel in ratio_channels:
        if channel not in channels:
            raise ValueError(f"The channel {channel} is not in the channels list {channels}.")

def _load_diff_arrays(img_paths: list[Path], mask_paths: list[Path], frame_idx: int, diff_channel_ratio: str | None, nchannels: int)-> tuple[np.ndarray, np.ndarray]:
    
    if diff_channel_ratio:
        # Unpack the channel ratio
        ratio_channels = diff_channel_ratio.split('/')
    
    mask_array = load_stack(mask_paths, frame_range=frame_idx, return_2D=True)
    if frame_idx == 0:
        # mask_array = load_stack(mask_paths, frame_range=frame_idx, return_2D=True)
        # Create an zero array with the same shape as the mask array, but with the number of channels, if 1 channel only squeeze will remove the extra dimension
        diff_array = np.squeeze(np.zeros(shape=(*mask_array.shape, nchannels))).astype(np.int16)
    else:
        # Load the mask array that includes the previous frame
        # mask_array = load_stack(mask_paths, frame_range=[frame_idx-1,frame_idx], return_2D=True)
        
        # Apply a logical_and operation to get the overlapping cells between the two frames
        # mask_array = np.where((mask_array[0]!=0) & (mask_array[1]!=0), mask_array[1], 0)
        
        # Load the image array
        if diff_channel_ratio:
            arr1 = load_stack(img_paths, channels=ratio_channels[0], frame_range=[frame_idx-1,frame_idx], return_2D=True).astype(np.float32)
            arr2 = load_stack(img_paths, channels=ratio_channels[1], frame_range=[frame_idx-1,frame_idx], return_2D=True).astype(np.float32)
            img_array = np.divide(arr1, arr2, out=np.zeros_like(arr1), where=arr2!=0, dtype=np.float32)
            # Replace the NaN or inf values with 0
            img_array[np.isinf(img_array) | np.isnan(img_array)] = 0
        else:
            img_array = load_stack(img_paths, frame_range=[frame_idx-1,frame_idx], return_2D=True)
            
        # Compute the difference
        diff_array = np.squeeze(np.diff(img_array.astype(np.int16), axis=0))
    return mask_array, diff_array

############### Custom properties functions ####################
def label_in(mask_region: np.ndarray, intensity_image: np.ndarray)-> bool:
    """Extra property function for the regionprops_table(). Look if masks in primary maks (aka: mask_region) are in the secondary masks (aka: intensity_image)."""
    
    
    return np.any(np.logical_and(mask_region,intensity_image)) 


