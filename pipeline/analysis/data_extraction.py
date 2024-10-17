from __future__ import annotations
from os import remove
from pathlib import Path
import pandas as pd
import numpy as np
from pipeline.utilities.data_utility import run_multithread, get_exp_props
from pipeline.utilities.pipeline_utility import PathType, progress_bar
from pipeline.analysis.extraction_core import extract_regionprops, PROPERTIES
from pipeline.analysis.imgs_loading import load_images_and_masks


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
    if num_chunks == 1:
        dfs.append(process_chunk(range(nframes), img_paths, masks_fold, do_diff, ref_masks_fold, exp_path, pixel_resolution))
    else: 
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
    img_masks = load_images_and_masks(chunk_frames, img_paths, masks_fold, ref_masks_fold, exp_path, pixel_resolution, channels, nframes)
    
    # Process the data
    dfs = process_arrays(chunk_frames, channels, *img_masks)

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
            df = extract_regionprops(None,**fixed_args)
            df.rename(columns=col_rename, inplace=True)
            dfs.append(df)
            continue
        
        # Else, if the mask_array is a time sequence
        lst_df = run_multithread(extract_regionprops, range(mask_array.shape[0]), fixed_args, add_lock=False)
        
        df = pd.concat(lst_df, ignore_index=True)
        df.rename(columns=col_rename, inplace=True)
        dfs.append(df)
    return dfs



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
    



