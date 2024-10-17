from __future__ import annotations
import numpy as np
import pandas as pd
from skimage.measure import regionprops_table

# Build-in properties for regionprops_table. NOTE: if modifying this list, make sure to update the _rename_columns() function.
PROPERTIES = ['area','centroid','intensity_mean',
                       'label','perimeter','slice','solidity']


def extract_regionprops(frame_idx: int | None, frame_vals: list[int], mask_array: np.ndarray, img_array: np.ndarray, mask_name: str, diff_array: np.ndarray | None, ref_masks: list[tuple[np.ndarray, str, float | None]] | None, sec_masks: list[tuple[np.ndarray, str]] | None)-> pd.DataFrame:
        """Core function to extract the regionprops from the mask_array and img_array. It will post-process the data if needed.
        
        Args:
            frame_idx (int | None): index of the frame to process. Set to None if the data is not a time sequence.
            
            frame_vals (list[int]): list of frame values for each chunk of the data
            
            mask_array (np.ndarray): array of masks
            
            img_array (np.ndarray): array of images
            
            mask_name (str): name of the mask
            
            do_diff (bool): whether to extract the differencial data in the regionprops
            
            ref_masks (list[tuple[np.ndarray, str, float | None]] | None): list of reference masks. Each tuples should contain the mask array, the name of that mask and the resolution to be converted to um, else will be in pixel. Defaults to None.
            
            sec_masks (list[tuple[np.ndarray, str]] | None): list of secondary masks. Each tuples should contain the mask array and the name of that mask. Defaults to None.
            
            Returns:
                pd.DataFrame: extracted data from the images."""
            
            
        # Extract the main regionprops
        if frame_idx is None:
            prop = regionprops_table(mask_array, img_array, properties=PROPERTIES, separator='_')
        else:
            prop = regionprops_table(mask_array[frame_idx], img_array[frame_idx], properties=PROPERTIES, separator='_')
        
        # List of processing functions
        processing_funcs = [
            (diff_props, diff_array),
            (ref_props, ref_masks),
            (sec_props, sec_masks)
        ]
        
        # Apply each processing function if the corresponding data is not None
        for func, data in processing_funcs:
            if data is not None:
                func(data, mask_array, frame_idx, prop)
        
        # Return the data as a dataframe       
        df = pd.DataFrame(prop)
        df['frame'] = frame_vals[frame_idx]+1
        df['mask_name'] = mask_name
        return df

def diff_props(diff_array: np.ndarray, mask_array: np.ndarray, frame_idx: int | None, prop: dict[str,float])-> None:
    """Function that will substract each frames with the previous frame to extract the difference in the regionprops."""
    
    # Get the number of channels
    nchannels = diff_array.shape[-1] if diff_array.ndim == 4 else 1
    
    # Rename the columns
    if nchannels > 1:
        col_rename = {f'intensity_mean_{i}': f'diff_intensity_mean_{i}' for i in range(nchannels)}
    else:
        col_rename = {'intensity_mean': 'diff_intensity_mean'}
    
    # Extract the regionprops
    if frame_idx is None:
        prop_diff = regionprops_table(mask_array,diff_array,properties=['intensity_mean'],separator='_')
    else:
        prop_diff = regionprops_table(mask_array[frame_idx],diff_array[frame_idx],properties=['intensity_mean'],separator='_')
    
    # Rename the props
    prop_diff = {col_rename[key]: value for key, value in prop_diff.items()}
    
    # Update the main properties with the difference
    prop.update(prop_diff)

def ref_props(ref_masks: list[tuple[np.ndarray, str, float | None]], mask_array: np.ndarray, frame_idx: int | None, prop: dict[str,float])-> None:
    """Extract the regionprops from the reference masks. The function will compute the distance transform value from the dmap mask of the centroid of the primary mask. The distance transform value will be added to the main properties."""
    
    
    for ref_mask, ref_name, resolution in ref_masks:
        # Check if the experiment is a time sequence
        if frame_idx is None:
            prop_ref = regionprops_table(mask_array,ref_mask,properties=['label'],separator='_',extra_properties=[dmap])
        else:
            prop_ref = regionprops_table(mask_array[frame_idx],ref_mask[frame_idx],properties=['label'],separator='_',extra_properties=[dmap])
        
        # Update the main properties with the dmap
        if resolution:
            prop[f'dmap_um_{ref_name}'] = prop_ref['dmap']*resolution
        else:
            prop[f'dmap_pixel_{ref_name}'] = prop_ref['dmap']

def sec_props(sec_masks: list[tuple[np.ndarray, str]], mask_array: np.ndarray, frame_idx: int | None, prop: dict[str, float])-> None:
    """Extract the regionprops from the secondary masks. The function will compute the overlap between the primary mask cells and the secondary masks cells and return a boolean value, whether the primary mask cells are in the secondary masks cells."""
    
    
    for sec_mask, sec_name in sec_masks:
        if frame_idx is None:
            prop_sec = regionprops_table(mask_array,sec_mask,properties=['intensity_max'],separator='_',extra_properties=[label_in])
        else:
            prop_sec = regionprops_table(mask_array[frame_idx],sec_mask[frame_idx],properties=['intensity_max'],separator='_',extra_properties=[label_in])
            
        # Update the main properties with the overlap
        prop[f'label_classification'] = [f"overlap with {sec_name}_{label}" if state else f"no overlap" for state, label in zip(prop_sec['label_in'], prop_sec['intensity_max'])]

############### Custom properties functions ####################
def dmap(mask_region: np.ndarray, intensity_image: np.ndarray)-> int:
    """Extra property function for the regionprops_table(). Extract the distance transform value from the dmap mask (i.e. intensity_image) of the centroid of the primary mask (i.e. mask_region)."""
        
        
    # Calculate the centroid coordinates as integers of mask
    y, x = np.nonzero(mask_region)
    y, x = int(np.mean(y)), int(np.mean(x))
    # Return the intensity at the centroid position
    return int(intensity_image[y,x])
    
def label_in(mask_region: np.ndarray, intensity_image: np.ndarray)-> bool:
    """Extra property function for the regionprops_table(). Look if masks in primary maks (aka: mask_region) are in the secondary masks (aka: intensity_image)."""
    
    
    return np.any(np.logical_and(mask_region,intensity_image)) 
