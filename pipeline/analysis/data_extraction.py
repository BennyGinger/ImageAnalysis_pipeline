from __future__ import annotations
from os import remove
from pathlib import Path
from typing import TypeVar
import pandas as pd
from tifffile import imwrite
import numpy as np
from skimage.measure import regionprops_table
from scipy.ndimage import distance_transform_edt
from pipeline.mask_transformation.utils import erode_masks
from pipeline.utilities.data_utility import run_multithread, load_stack, get_exp_props
from pipeline.utilities.pipeline_utility import PathType, progress_bar
# Custom variable type
T = TypeVar('T')

# Build-in properties for regionprops_table. NOTE: if modifying this list, make sure to update the _rename_columns() function.
PROPERTIES = ['area','centroid','intensity_mean',
                       'label','perimeter','slice','solidity']

########################### Main functions ###########################
def extract_data(img_paths: list[PathType], exp_path: Path, masks_fold: list[str], do_diff: bool, ref_masks_fold: list[str] | None, pixel_resolution: float | None=None, num_chunks: int=1, overwrite: bool=False)-> pd.DataFrame:
    """Extract images properties using the skimage.measure.regionprops_table with provided masks. The image can be processed either as a time sequence (F) or as a single frame. The mask must have must have the same shape as the image. Additionally, if a channel dim (C) is provided for the image, the mean intensity of each channel will be extracted. The expected shapes are ([F],Y,X,[C]) for the image and ([F],Y,X) for the mask, with [F] and [C] being optional. The data will be returned as a pandas.DataFrame, which willl also be saved at the save_path provided as a csv file named 'regionprops.csv'.
    Reference masks can also be provided to extract the distance transform from the centroid of the primary mask. The distance transform will be added to the main properties as 'dmap_um_{ref_name}' or 'dmap_pixel_{ref_name}' if the pixel resolution is not provided. The secondary masks can also be provided to check if the primary mask cells overlap with the secondary masks cells. The overlap will be added to the main properties as '{sec_name}{label}_overlap' or 'no_overalp'. If masks_folders contain a folder with several channels, the function will automatically pair the channels to extract the secondary masks.
    The differencial data can also be extracted by setting the do_diff to True. The function will substract each frames with the previous frame to extract the difference in the regionprops. The differencial data will be added to the main properties as 'diff_int_mean_{channel}'.
    
    Args:
        img_paths (list[Path]): list of paths to the images
        
        exp_path (Path): path to the experiment folder
        
        masks_fold (list[str]): list of folders names containing the masks
        
        do_diff (bool): whether to extract the differencial data in the regionprops
        
        ref_masks_fold (list[str]): list of folders names containing the reference masks
        
        pixel_resolution (float | None): pixel resolution of the image to convert the distance into um. If set to None, the distance transform will be in pixel. Defaults to None.
        
        num_chunks (int): number of chunks to process the data. Defaults to 1.
        
        overwrite (bool): whether to overwrite the data if it already exists. Defaults to False.
        
    Returns:
        pd.DataFrame: extracted data from the images.
    
    """
    
    
    # Check if the data has already been extracted
    csv_file = exp_path.joinpath("regionprops.csv")
    print(f" --> Extracting data from \033[94m{csv_file}\033[0m")
    if csv_file.exists() and not overwrite:
        print(f"  ---> Data has already been extracted. Loading data from \033[94m{csv_file}\033[0m")
        return pd.read_csv(csv_file)
    else:
        # If overwrite and the file exists, remove the file
        remove(csv_file) if csv_file.exists() else None
    
    # Get the number of frames
    nframes = get_exp_props(img_paths)[2]
    
    # Define the chunk size
    chunk_indinces = np.linspace(0, nframes, num_chunks + 1, dtype=int)
    
    # Process the data in chunks
    dfs = []
    for i in progress_bar(range(num_chunks), desc="Chunk Data Extraction"):
        chunk_frames = range(chunk_indinces[i], chunk_indinces[i+1])
        dfs.append(process_chunk(chunk_frames, img_paths, masks_fold, do_diff, ref_masks_fold, exp_path, pixel_resolution))
    
    # Concatenate the dataframes
    df = pd.concat(dfs, ignore_index=True)
    # Sort the dataframe
    df = df.sort_values(by=['mask_name','frame','cell_label'])
    df.to_csv(csv_file, index=False)
    return df
    
    
############################# Helper functions #############################
def process_chunk(chunk_frames: range, img_paths: list[Path], masks_fold: list[str], do_diff: bool, ref_masks_fold: list[str] | None, exp_path: Path, pixel_resolution: float | None)-> pd.DataFrame:
    """_summary_

    Args:
        chunk_frames (np.ndarray): array of frame indices to process
        img_paths (list[Path]): list of paths to the images
        save_path (Path): path to save the extracted data
        ref_masks_fold (list[str]): list of folders names containing the reference masks

    Returns:
        pd.DataFrame: dataframe of the extracted data from the chunk of frames
    """
    

    # Load images
    channels, _, nframes, _ = get_exp_props(img_paths)
    img_array = load_stack(img_paths, channels, chunk_frames, True)
    
    # Load ref masks, if provided, as ref_masks_fold can be an empty list
    ref_masks = load_reference_masks(chunk_frames, ref_masks_fold, exp_path, pixel_resolution)

    # Load masks
    pair_arrays = generate_mask_pairs(chunk_frames, masks_fold, exp_path)

    # Load diff masks
    if nframes == 1:
        do_diff = False
    diff_array = load_diff_masks(img_array) if do_diff else None
    
    # Process the data
    dfs = process_arrays(chunk_frames, channels, img_array, diff_array, ref_masks, pair_arrays)

    # Concatenate the dataframes
    return pd.concat(dfs, ignore_index=True)

def process_arrays(chunk_frames: range, channels: list[str], img_array: np.ndarray, diff_array: np.ndarray | None, ref_masks: list[tuple[np.ndarray, str, float | None]] |None, pair_arrays: list[tuple[tuple[np.ndarray, str], list[tuple[np.ndarray, str]] | None]])-> list[pd.DataFrame]:
    dfs = []
    for mask_tup, sec_masks in pair_arrays:
        mask_array, mask_name = mask_tup
        col_rename = _rename_columns(channels, diff_array)
        fixed_args = {'frame_vals':list(chunk_frames),
                      'mask_array':mask_array,
                      'img_array':img_array,
                      'mask_name':mask_name,
                      'diff_array':diff_array,
                      'ref_masks':ref_masks,
                      'sec_masks':sec_masks}
        
        if mask_array.ndim ==2:
            df = _extract_regionprops(None,**fixed_args)
            df.rename(columns=col_rename, inplace=True)
            dfs.append(df)
            continue
        
        # Else, if the mask_array is a time sequence
        lst_df = run_multithread(_extract_regionprops, range(mask_array.shape[0]), fixed_args)
        
        df = pd.concat(lst_df, ignore_index=True)
        df.rename(columns=col_rename, inplace=True)
        dfs.append(df)
    return dfs

def generate_mask_pairs(chunk_frames: range, masks_fold: list[str], exp_path: Path)-> list[tuple[tuple[np.ndarray, str], list[tuple[np.ndarray, str]] | None]]: 
    pair_arrays: list[tuple[tuple[np.ndarray, str], list[tuple[np.ndarray, str]] | None]] = []
    for fold in masks_fold:
        mask_path = exp_path.joinpath(fold)
        process_name = fold.split('_', maxsplit=1)[-1].lower()
        mask_files = sorted(mask_path.glob('*.tif'))
        mask_channels = get_exp_props(mask_files)[0]
        if mask_channels == 1:
            pair_channels = [(mask_channels[0], None)]
        else:
            pair_channels = make_pairs(mask_channels)
        
        # Erode secondary masks
        for chan, sec_channels in pair_channels:
            primary_mask = load_stack(mask_files, chan, chunk_frames, True)
            primary_name = f"{process_name}_{chan}"
            if sec_channels is not None:
                sec_masks = [load_stack(mask_files, sec_chan, chunk_frames, True) for sec_chan in sec_channels]
                sec_masks = [erode_masks(sec_mask) for sec_mask in sec_masks]
                sec_masks = list(zip(sec_masks, sec_channels))
            else:
                sec_masks = None
            pair_arrays.append(((primary_mask, primary_name), sec_masks))
    return pair_arrays

def load_reference_masks(chunk_frames: range, ref_masks_fold: list[str] | None, exp_path: Path, pixel_resolution: float | None)-> list[tuple[np.ndarray, str, float | None]] | None:
    if ref_masks_fold is None:
        return None
    
    ref_masks = []
    for ref_fold in ref_masks_fold:
        ref_path = exp_path.joinpath(ref_fold)
        ref_files = sorted(ref_path.glob('*.tif'))
        ref_array = load_stack(ref_files, frame_range=chunk_frames, return_2D=True)
        ref_array = _dist_transform(ref_array)
        ref_name = ref_fold.split('_')[-1]
        ref_masks.append((ref_array, ref_name, pixel_resolution))
    return ref_masks

def load_diff_masks(img_array: np.ndarray)-> np.ndarray:
    # Extract the differencial data, as int16, to keep the negative values
    img_array_diff = np.concatenate((img_array[0:1], np.diff(img_array.astype(np.int16), axis=0)), axis=0)
    return img_array_diff

def _extract_regionprops(frame_idx: int | None, frame_vals: list[int], mask_array: np.ndarray, img_array: np.ndarray, mask_name: str, diff_array: np.ndarray | None, ref_masks: list[tuple[np.ndarray, str, float | None]] | None, sec_masks: list[tuple[np.ndarray, str]] | None,**kwargs)-> pd.DataFrame:
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
        
        # Extract the differencial props
        if diff_array is not None:
            diff_props(frame_idx, mask_array, diff_array, prop)
        
        # Extract the dmap from the reference masks
        if ref_masks is not None:
            ref_props(ref_masks, mask_array, frame_idx, prop)
        
        # Assess the classification of each channels
        if sec_masks is not None:
            sec_props(sec_masks, mask_array, frame_idx, prop)
        
        # Return the data as a dataframe       
        df = pd.DataFrame(prop)
        df['frame'] = frame_vals[frame_idx]+1
        df['mask_name'] = mask_name
        return df

def diff_props(frame_idx: int | None, mask_array: np.ndarray, diff_array: np.ndarray, prop: dict[str,float])-> None:
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

def _rename_columns(channels: list[str], diff_array: np.ndarray | None)-> dict[str, str]:
    """Function to rename the columns of the regionprops_table output. The columns will be renamed
    with the channels names."""
    
    
    # Setup the column renaming
    col_rename = {}
    if 'centroid' in PROPERTIES:
        col_rename.update({'centroid_0':'centroid_y','centroid_1':'centroid_x'})
    if 'label' in PROPERTIES:
        col_rename.update({'label':'cell_label'})
    
    # If the img_array has a channel dimension
    if 'intensity_mean' in PROPERTIES:
        if len(channels) > 1: 
            col_rename = {**col_rename, **{f'intensity_mean_{i}': f'intensity_mean_{channels[i]}' for i in range(len(channels))}}
        else:
            col_rename['intensity_mean'] = f'intensity_mean_{channels[0]}'
    
    # If the difference is computed
    if diff_array is not None:
        if len(channels) > 1:
            col_rename = {**col_rename, **{f'diff_intensity_mean_{i}': f'diff_int_mean_{channels[i]}' for i in range(len(channels))}}
        else:
            col_rename['diff_intensity_mean'] = f'diff_int_mean_{channels[0]}'
    
    return col_rename

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

def _dist_transform(mask: np.ndarray)-> np.ndarray:
    """Apply the distance transform to the mask."""
    
    
    return distance_transform_edt(np.logical_not(mask))    

def make_pairs(lst: list[T])-> list[tuple[T, list[T]]]:
    """Make pairs of elements from a list. For example, if the list is [1,2,3], the output will be [(1,[2,3]),(2,[1,3]),(3,[1,2])]."""
    
    pairs = []
    for i, element in enumerate(lst):
        others = lst[:i] + lst[i+1:]
        pairs.append((element, others))
    return pairs

# # # # # # # # # Test
if __name__ == "__main__":
    import time
    from os import listdir
    from os.path import join 
    from tifffile import imread, imwrite
    from pipeline.utilities.data_utility import load_stack
    from scipy.ndimage import distance_transform_edt
    
    img_folder = Path("/home/Test_images/nd2/Run4/c4z1t91v1_s1/Images_Registered")
    img_paths = sorted(Path(img_folder).glob("*.tif"))
    mask_folder = ['Masks_IoU_Track']
    
    start = time.time()
    # extract props
    master_df = extract_data(img_paths=img_paths,
                             exp_path=img_folder.parent,
                             masks_fold=mask_folder,
                             do_diff=True,
                             ref_masks_fold=None,
                             pixel_resolution=None,
                             num_chunks=1,
                             overwrite=True)
    end = time.time()
    print(f"Processing time: {end-start}")
    



