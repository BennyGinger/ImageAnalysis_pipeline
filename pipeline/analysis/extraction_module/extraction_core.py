from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from tifffile import imread
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
def extract_regionprops(frame_idx: int, mask_data: dict[str, list[Path]], img_paths: list[Path], do_diff: bool, ref_data: dict[str, list[Path]] | None, class_data: dict[str, list[Path]] | None, diff_channel_ratio: str | None = None, ref_resolution: float | None = None, do_compart: bool = False)-> pd.DataFrame:
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
        prop = base_props(frame_idx, img_paths, mask_paths, do_diff, diff_channel_ratio)
        
        if do_compart:
            comp_props(frame_idx, img_paths, mask_paths, prop)
        
        if ref_data:
            ref_props(ref_data, ref_resolution, mask_paths, frame_idx, prop, do_diff)
        
        if class_data:
            class_props(class_data, mask_paths, frame_idx, prop, do_diff)
        
        # Return the data as a dataframe       
        df = pd.DataFrame(prop)
        df['frame'] = frame_idx+1
        df['mask_name'] = mask_name
        return df


################# Processing functions ####################
def base_props(frame_idx: int, img_paths: list[Path], mask_paths: list[Path], do_diff: bool, diff_channel_ratio: str | None = None)-> dict[str,any]:
    # Get img properties
    channels, _, nframes, _ = get_exp_props(img_paths)
    
    if diff_channel_ratio:
        # Validate the ratio
        _validate_channel_ratio(channels, diff_channel_ratio)
        channels = [diff_channel_ratio]
    
    # Load the mask and image arrays
    if do_diff and nframes > 1:
        img_array = load_diff_img_arrays(img_paths, frame_idx, channels)
        mask_array = load_diff_mask_array(mask_paths, frame_idx)
        
    else:
        img_array = load_stack(img_paths, frame_range=frame_idx, return_2D=True)
        mask_array = load_stack(mask_paths, frame_range=frame_idx, return_2D=True)
    
    # Extract the base regionprops
    return regionprops_table(mask_array, img_array, properties=PROPERTIES, separator='_')

def comp_props(frame_idx: int, img_paths: list[Path], mask_paths: list[Path], prop: dict[str, any])-> None:
    # Load image and masks arrays
    img_array, mb_array, cyto_array = _load_compart_arrays(frame_idx, img_paths, mask_paths)
    
    # Find missing masks, if any
    missing_masks = _missing_masks_indexes(mb_array, cyto_array)
    
    # Extract the regionprops of mb and cyto compartments
    _extract_compart(prop, mb_array, img_array, 'mb')
    _extract_compart(prop, cyto_array, img_array, 'cyto', missing_masks)

def ref_props(ref_data: dict[str, list[Path]], resolution: float | None, mask_paths: list[Path], frame_idx: int, prop: dict[str,any], do_diff: bool)-> None:
    """Extract the regionprops from the reference masks. The function will compute the distance transform value from the dmap mask of the centroid of the primary mask. The distance transform value will be added to the main properties."""
    # Load the mask array
    if do_diff:
        mask_array = load_diff_mask_array(mask_paths, frame_idx)
    else:
        mask_array = load_stack(mask_paths, frame_range=frame_idx, return_2D=True)
    
    # Get the unit of the mask
    unit_name = 'um' if resolution else 'pixel'
    
    # Extract the reference masks
    for ref_name, ref_paths in ref_data.items():
        # Apply the distance transform to the reference array
        ref_array = load_stack(ref_paths, frame_range=frame_idx, return_2D=True)
        
        # Update the main properties with the dmap
        prop[f'dmap_{unit_name}_{ref_name}'] = _get_min_distance(mask_array,ref_array,resolution)

def class_props(class_data: dict[str, list[Path]], mask_paths: list[Path], frame_idx: int | None, prop: dict[str, any], do_diff: bool)-> None:
    """Extract the regionprops from the secondary masks. The function will compute the overlap between the primary mask cells and the secondary masks cells and return a boolean value, whether the primary mask cells are in the secondary masks cells."""
    
    # Load the mask array
    if do_diff:
        mask_array = load_diff_mask_array(mask_paths, frame_idx)
    else:
        mask_array = load_stack(mask_paths, frame_range=frame_idx, return_2D=True)
    
    for class_name, class_paths in class_data.items():
        # Load the classification array
        class_array = load_diff_mask_array(class_paths, frame_idx, class_name)
        # Erode the classification masks
        class_array = erode_masks(class_array)
        
        # Extract the regionprops
        prop_sec = regionprops_table(mask_array, class_array, properties=['intensity_max'], separator='_', extra_properties=[label_in])
            
        # Update the main properties with the overlap
        prop[f'Label_classification'] = [f"{class_name}_{label} overlaps" if state else f"no overlap" for state, label in zip(prop_sec['label_in'], prop_sec['intensity_max'])]


################# Helper functions ####################
def _load_compart_arrays(frame_idx: int, img_paths: list[Path], mask_paths: list[Path])-> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask_chan = get_exp_props(mask_paths)[0][0]
    exp_path = mask_paths[0].parent.parent
    compart_folder = exp_path.joinpath('Masks_Compartment')
    compart_files = [file for file in list(compart_folder.iterdir()) if mask_chan in file.name]
    cyto_files = sorted([file for file in compart_files if 'cyto' in file.name])
    mb_files = sorted([file for file in compart_files if 'mb' in file.name])
    
    # Load image and masks arrays
    img_array = load_stack(img_paths, frame_range=frame_idx, return_2D=True)
    mb_array = load_stack(mb_files, frame_range=frame_idx, return_2D=True)
    cyto_array = load_stack(cyto_files, frame_range=frame_idx, return_2D=True)
    return img_array, mb_array, cyto_array

def _missing_masks_indexes(mb_array: np.ndarray, cyto_array: np.ndarray)-> list[int]:
    unique_masks = np.setdiff1d(np.unique(mb_array)[1:], np.unique(cyto_array)[1:])
    masks_indexes = np.where(np.isin(np.unique(mb_array)[1:], unique_masks))[0]
    if masks_indexes.size == 0:
        return []
    return list(masks_indexes)
    
def _extract_compart(prop: dict, mask_array: np.ndarray, img_array: np.ndarray, compart_name: str, missing_masks_indexes: list[int] | None = None)-> None:
    
    prop_temp = regionprops_table(mask_array, img_array, properties=['intensity_mean'], separator='_')
    prop_renamed = {f'{compart_name}_{key}': value for key, value in prop_temp.items()}
    
    # Add nan values to any missing cyto masks
    if missing_masks_indexes:
        for idx in missing_masks_indexes:
            try:
                prop_renamed = {key: np.insert(value, idx, np.nan) for key, value in prop_renamed.items()}
            except IndexError:
                prop_renamed = {key: np.append(value, np.nan) for key, value in prop_renamed.items()}
    
    prop.update(prop_renamed)

def _get_min_distance(mask: np.ndarray, ref_array: np.ndarray, resolution: float | None)-> list[np.ndarray]:
    # Get the stacked coordinates of the mask and the reference array
    mask_coords = np.column_stack(np.where(mask != 0))
    ref_coords = np.column_stack(np.where(ref_array != 0))
    
    # Get the uniques labels
    objects_ids = np.unique(mask)[1:]
    
    # Get the centroids of the masks
    mask_centroids = np.array([np.mean(mask_coords[mask[mask_coords[:,0],mask_coords[:,1]] == obj], axis=0) for obj in objects_ids])
    if mask_centroids.size == 0:
        return []
    # Compute the minimum distance between each mask and the reference array
    dist_array = np.min(distance.cdist(mask_centroids, ref_coords), axis=1)
    if resolution:
        dist_array = dist_array*resolution
    return list(dist_array)

def _validate_channel_ratio(channels: list[str], ratio: str)-> None:
    ratio_channels = ratio.split('/')
    
    if len(ratio_channels) != 2:
        raise ValueError("The ratio should be in the form 'channel1/channel2'")
    
    for channel in ratio_channels:
        if channel not in channels:
            raise ValueError(f"The channel {channel} is not in the channels list {channels}.")

def load_diff_mask_array(mask_paths: list[Path], frame_idx: int, channel: str | None = None)-> np.ndarray:
    
    if frame_idx == 0:
        return load_stack(mask_paths, channels=channel, frame_range=frame_idx, return_2D=True)
    
    # Load the mask array that includes the previous frame
    mask_array = load_stack(mask_paths, channels=channel, frame_range=[frame_idx-1,frame_idx], return_2D=True)
    
    # Apply a logical_and operation to get the overlapping cells between the two frames
    return np.where((mask_array[0]!=0) & (mask_array[1]!=0), mask_array[1], 0)

def load_diff_img_arrays(img_paths: list[Path], frame_idx: int, channels: list[str])-> np.ndarray:
    
    nchannels = len(channels)
    ratio_channels = None
    
    # Unpack the channel ratio, if exists
    if nchannels == 1 and '/' in channels[0]:
        ratio_channels = channels[0].split('/')
    
    # Load the image array
    if frame_idx == 0:
        # Create an zero array with the same shape as the mask array, but with the number of channels, if 1 channel only squeeze will remove the extra dimension
        img_shape = imread(img_paths[0]).shape
        return np.squeeze(np.zeros(shape=(*img_shape, nchannels))).astype(np.int16)
    
    if ratio_channels:
        # Compute the ratio between the two channels
        arr1 = load_stack(img_paths, channels=ratio_channels[0], frame_range=[frame_idx-1,frame_idx], return_2D=True).astype(np.float32)
        arr2 = load_stack(img_paths, channels=ratio_channels[1], frame_range=[frame_idx-1,frame_idx], return_2D=True).astype(np.float32)
        img_array = np.divide(arr1, arr2, out=np.zeros_like(arr1), where=arr2!=0, dtype=np.float32)
        # Replace the NaN or inf values with 0
        img_array[np.isinf(img_array) | np.isnan(img_array)] = 0
    else:
        img_array = load_stack(img_paths, frame_range=[frame_idx-1,frame_idx], return_2D=True)
        
    # Compute the difference
    return np.squeeze(np.diff(img_array.astype(np.int16), axis=0))


############### Custom properties functions ####################
def label_in(mask_region: np.ndarray, intensity_image: np.ndarray)-> bool:
    """Extra property function for the regionprops_table(). Look if masks in primary maks (aka: mask_region) are in the secondary masks (aka: intensity_image)."""
    return np.any(np.logical_and(mask_region,intensity_image)) 


