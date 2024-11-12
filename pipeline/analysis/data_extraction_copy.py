from __future__ import annotations
from os import remove
from pathlib import Path
import pandas as pd
from pipeline.utilities.data_utility import run_multithread, get_exp_props
from pipeline.analysis.extraction_utilities.extraction_core_copy import extract_regionprops, PROPERTIES
from typing import TypeVar

# Custom variable type
T = TypeVar('T')

########################### Main functions ###########################
def extract_data(img_paths: list[Path], exp_path: Path, masks_fold: list[str], do_diff: bool, ref_masks_fold: list[str] | None, pixel_resolution: float | None=None, diff_channel_ratio: str | None = None, overwrite: bool=False)-> pd.DataFrame:
    
    # Check if the data has already been extracted
    csv_file = exp_path.joinpath("regionprops.csv")
    print(f" --> Extracting data from \033[94m{csv_file}\033[0m")
    if csv_file.exists() and not overwrite:
        print(f"  ---> Data has already been extracted. Loading data from \033[94m{csv_file}\033[0m")
        return pd.read_csv(csv_file)
    else:
        # If overwrite and the file exists, remove the file
        remove(csv_file) if csv_file.exists() else None
    
    # Get the experiment properties
    channels, _, nframes, _ = get_exp_props(img_paths)
    
    # Load the mask data, as a list of tuples with the mask data (dict) and the class data (dict | None)
    paired_masks = _group_mask_data(exp_path, masks_fold)
    
    # Load the reference paths
    ref_data = _load_reference_data(exp_path, ref_masks_fold)
    
    # Process the data
    df_lst = []
    for mask_data, class_data in paired_masks:
        col_rename = _rename_columns(channels)
        fixed_args = {'mask_data': mask_data, 
                      'img_paths': img_paths, 
                      'do_diff': do_diff, 
                      'ref_data': ref_data, 
                      'class_data': class_data,
                      'diff_channel_ratio': diff_channel_ratio,
                      'ref_resolution': pixel_resolution}
        
        if nframes == 1:
            df = extract_regionprops(0, **fixed_args)
            df.rename(columns=col_rename, inplace=True)
        
        else:
            dfs = run_multithread(extract_regionprops, range(nframes), fixed_args, add_lock=False)
            # dfs = run_multithread(extract_regionprops, [18], fixed_args, add_lock=False)
            df = pd.concat(dfs, ignore_index=True)
            df.rename(columns=col_rename, inplace=True)
        
        df_lst.append(df)
    
    # Concatenate the dataframes
    region_df = pd.concat(df_lst, ignore_index=True)
    region_df = region_df.sort_values(by=['mask_name','frame','cell_label'])
    region_df.to_csv(csv_file, index=False)
    return region_df


############################# Helper functions #############################
def _make_pairs(lst: list[T])-> list[tuple[T, list[T]]]:
    """Make pairs of elements from a list. For example, if the list is [1,2,3], the output will be [(1,[2,3]),(2,[1,3]),(3,[1,2])]."""
    if len(lst) == 1:
        return [(lst[0], None)]
    
    pairs = []
    for i, element in enumerate(lst):
        others = lst[:i] + lst[i+1:]
        pairs.append((element, others))
    return pairs

def _rename_columns(channels: list[str])-> dict[str, str]:
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
    
    return col_rename

def _group_mask_data(exp_path: Path, masks_fold: list[str])-> list[tuple[dict[str, list[Path]], dict[str, list[Path]] | None]]:
    paired_masks = []
    for fold in masks_fold:
        mask_path = exp_path.joinpath(fold)
        process_name = fold.split('_', maxsplit=1)[-1].lower()
        mask_files = sorted(mask_path.glob('*.tif'))
        mask_channels = get_exp_props(mask_files)[0]
        pair_channels = _make_pairs(mask_channels)
        
        for chan, class_chan in pair_channels:
            mask_data = {f"{process_name}_{chan}": [files for files in mask_files if chan in files.name]}
            if class_chan:
                class_data = {chan: [files for files in mask_files if chan in files.name] for chan in class_chan}
            else:
                class_data = None
            paired_masks.append((mask_data, class_data))
    return paired_masks

def _load_reference_data(exp_path: Path, ref_masks_fold: list[str])-> dict[str, list[Path]] | None:
    if ref_masks_fold:
        ref_data = {}
        for ref_fold in ref_masks_fold:
            ref_path = exp_path.joinpath(ref_fold)
            ref_files = sorted(ref_path.glob('*.tif'))
            ref_data[ref_fold] = ref_files
    else:
        ref_data = None
    return ref_data






# # # # # # # # # Test
if __name__ == "__main__":
    import time
    
    img_folder = Path("/home/Test_images/dia_fish/newtest/c1172-GCaMP-15%_Hypo-1-MaxIP_s1/Images_Registered")
    img_paths = sorted(Path(img_folder).glob("*.tif"))
    mask_folder = ['Masks_Cellpose']
    
    start = time.time()
    # extract props
    master_df = extract_data(img_paths=img_paths,
                             exp_path=img_folder.parent,
                             masks_fold=mask_folder,
                             do_diff=True,
                             ref_masks_fold=['Masks_laser'],
                             pixel_resolution=None,
                             diff_channel_ratio=None,
                             overwrite=True)
    end = time.time()
    print(f"Processing time: {end-start}")
    



