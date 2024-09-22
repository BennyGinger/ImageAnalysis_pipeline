from __future__ import annotations
from os import remove
from pathlib import Path
from typing import TypeVar
import pandas as pd
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
def extract_data(img_paths: list[PathType], exp_path: Path, masks_fold: list[str], ref_masks_fold: list[str], pixel_resolution: float | None=None, num_chunks: int=1, overwrite: bool=False)-> pd.DataFrame:
    """Extract images properties using the skimage.measure.regionprops_table with provided masks. The image can be processed either as a time sequence (F) or as a single frame. The mask must have must have the same shape as the image. Additionally, if a channel dim (C) is provided for the image, the mean intensity of each channel will be extracted. The expected shapes are ([F],Y,X,[C]) for the image and ([F],Y,X) for the mask, with [F] and [C] being optional. The data will be returned as a pandas.DataFrame, which willl also be saved at the save_path provided as a csv file named 'regionprops.csv'.
    Reference masks can also be provided to extract the distance transform from the centroid of the primary mask. The distance transform will be added to the main properties as 'dmap_um_{ref_name}' or 'dmap_pixel_{ref_name}' if the pixel resolution is not provided. The secondary masks can also be provided to check if the primary mask cells overlap with the secondary masks cells. The overlap will be added to the main properties as 'label_{sec_name}_positive'. If masks_folders contain a folder with several channels, the function will automatically pair the channels to extract the secondary masks.
    
    Args:
        img_paths (list[Path]): list of paths to the images
        
        exp_path (Path): path to the experiment folder
        
        masks_fold (list[str]): list of folders names containing the masks
        
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
        dfs.append(process_chunk(chunk_frames, img_paths, masks_fold, ref_masks_fold, exp_path, pixel_resolution))
    
    # Concatenate the dataframes
    df = pd.concat(dfs, ignore_index=True)
    # Sort the dataframe
    df = df.sort_values(by=['mask_name','frame','cell_label'])
    df.to_csv(csv_file, index=False)
    return df
    
    
############################# Helper functions #############################
def process_chunk(chunk_frames: range, img_paths: list[Path], masks_fold: list[str], ref_masks_fold: list[str], exp_path: Path, pixel_resolution: float | None)-> pd.DataFrame:
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
    channels = get_exp_props(img_paths)[0]
    img_array = load_stack(img_paths, channels, chunk_frames, True)
    
    # Load ref masks, if provided, as ref_masks_fold can be an empty list
    ref_masks = load_reference_masks(chunk_frames, ref_masks_fold, exp_path, pixel_resolution)

    # Load masks
    pair_arrays = generate_mask_pairs(chunk_frames, masks_fold, exp_path)
            
    # Process the data
    dfs = process_arrays(chunk_frames, channels, img_array, ref_masks, pair_arrays)

    # Concatenate the dataframes
    return pd.concat(dfs, ignore_index=True)

def process_arrays(chunk_frames: range, channels: list[str], img_array: np.ndarray, ref_masks: list[tuple[np.ndarray, str, float | None]], pair_arrays):
    dfs = []
    for mask_tup, sec_masks in pair_arrays:
        mask_array, mask_name = mask_tup
        col_rename = _rename_columns(img_array.ndim, mask_array.ndim, channels)
        fixed_args = {'frame_vals':list(chunk_frames),
                      'mask_array':mask_array,
                      'img_array':img_array,
                      'mask_name':mask_name,
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

def load_reference_masks(chunk_frames: range, ref_masks_fold: list[str], exp_path: Path, pixel_resolution: float | None)-> list[tuple[np.ndarray, str, float | None]]:
    ref_masks = []
    for ref_fold in ref_masks_fold:
        ref_path = exp_path.joinpath(ref_fold)
        ref_files = sorted(ref_path.glob('*.tif'))
        ref_array = load_stack(ref_files, frame_range=chunk_frames, return_2D=True)
        ref_array = _dist_transform(ref_array)
        ref_name = ref_fold.split('_')[-1]
        ref_masks.append((ref_array, ref_name, pixel_resolution))
    return ref_masks

def _extract_regionprops(frame_idx: int | None, frame_vals: list[int], mask_array: np.ndarray, img_array: np.ndarray, mask_name: str, ref_masks: list[tuple[np.ndarray, str, float | None]] | None=None, sec_masks: list[tuple[np.ndarray, str]] | None=None,**kwargs)-> pd.DataFrame:
        """Function to extract the regionprops from the mask_array and img_array. The function will extract the properties defined in the PROPERTIES list. If the ref_masks and/or the sec_maks are provided, the function will extract the dmap from the reference masks and/or whether the cells in pramary masks overlap with cells of the secondary masks. The extracted data will be returned as a pandas.DataFrame.
        
        Args:
            frame (int): The frame index to extract the data from.
            
            mask_array ([[F],Y,X], np.ndarray): The mask array to extract the regionprops from. Frame dim is optional.
            
            img_array ([[F],Y,X,[C]], np.ndarray): The image array to extract the mean intensities from. Frame and channel dim are optional.
            
            ref_masks (list[tuple[np.ndarray,str,float|None]], optional): list of tuples containing ref array, ref name and resolution of the image to generate the distance transform. Defaults to None.
            
            sec_masks (list[tuple[np.ndarray,str]], optional): list of tuples containing sec array and sec name to check if the primary mask cells are in the secondary masks. Defaults to None.
            
            kwargs: Additional arguments to pass to the regionprops_table function. Notably a lock to run this function in multithreading. Not implemented yet.
            
        Returns:
            pd.DataFrame: The extracted regionprops data.
            """
            
            
        ## Extract the main regionprops, meaning the intensity value for all the channels of the given mask
        # Check if mask is a time sequence
        if mask_array.ndim == 2:
            prop = regionprops_table(mask_array,img_array,properties=PROPERTIES,separator='_')
        else:
            prop = regionprops_table(mask_array[frame_idx],img_array[frame_idx],properties=PROPERTIES,separator='_')
        
        # Extract the dmap from the reference masks
        if ref_masks:
            ref_props(ref_masks,mask_array,frame_idx,prop)
        
        if sec_masks:
            sec_props(sec_masks,mask_array,frame_idx,prop)
                
        df = pd.DataFrame(prop)
        df['frame'] = frame_vals[frame_idx]+1
        df['mask_name'] = mask_name
        return df

def ref_props(ref_masks: list[tuple[np.ndarray, str, float | None]], mask_array: np.ndarray, frame_idx: int, prop: dict[str,float])-> None:
    """Extract the regionprops from the reference masks. The function will compute the distance transform value from the dmap mask of the centroid of the primary mask. The distance transform value will be added to the main properties."""
    
    
    for ref_mask, ref_name, resolution in ref_masks:
        # Check if the experiment is a time sequence
        if mask_array.ndim == 2:
            prop_ref = regionprops_table(mask_array,ref_mask,properties=['label'],separator='_',extra_properties=[dmap])
        else:
            prop_ref = regionprops_table(mask_array[frame_idx],ref_mask[frame_idx],properties=['label'],separator='_',extra_properties=[dmap])
        
        # Update the main properties with the dmap
        if resolution:
            prop[f'dmap_um_{ref_name}'] = prop_ref['dmap']*resolution
        else:
            prop[f'dmap_pixel_{ref_name}'] = prop_ref['dmap']

def sec_props(sec_masks: list[tuple[np.ndarray, str]], mask_array: np.ndarray, frame_idx: int, prop: dict[str, float])-> None:
    """Extract the regionprops from the secondary masks. The function will compute the overlap between the primary mask cells and the secondary masks cells and return a boolean value, whether the primary mask cells are in the secondary masks cells."""
    
    
    for sec_mask, sec_name in sec_masks:
        if mask_array.ndim == 2:
            prop_sec = regionprops_table(mask_array,sec_mask,properties=['intensity_max'],separator='_',extra_properties=[label_in])
        else:
            prop_sec = regionprops_table(mask_array[frame_idx],sec_mask[frame_idx],properties=['intensity_max'],separator='_',extra_properties=[label_in])
            
        # Update the main properties with the overlap
        prop[f'label_classification'] = [f"overlap with {sec_name}_{label}" if state else f"no overlap" for state, label in zip(prop_sec['label_in'], prop_sec['intensity_max'])]

def _rename_columns(img_dim: int, mask_dim: int, channels: str | list[str] | None)-> dict[str, str]:
    """Function to rename the columns of the regionprops_table output. The columns will be renamed
    with the channels names."""
    
    
    # Setup channels
    if channels is None:
        channels = [f"C{str(i+1)}" for i in range(img_dim)]
    
    if isinstance(channels, str):
        channels = [channels]
    
    # Setup the column renaming
    col_rename = {}
    if 'centroid' in PROPERTIES:
        col_rename.update({'centroid_0':'centroid_y','centroid_1':'centroid_x'})
    if 'label' in PROPERTIES:
        col_rename.update({'label':'cell_label'})
    
    # If the img_array has a channel dimension
    if 'intensity_mean' in PROPERTIES:
        if img_dim != mask_dim: 
            col_rename = {**col_rename, **{f'intensity_mean_{i}': f'intensity_mean_{channels[i]}' for i in range(len(channels))}}
        else:
            col_rename['intensity_mean'] = f'intensity_mean_{channels[0]}'
    
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
    
    img_folder = Path("/home/Test_images/dia_fish/newtest/c1172-GCaMP-15%_Hypo-1-MaxIP_s1/Images_Registered")
    img_paths = sorted(Path(img_folder).glob("*.tif"))
    
    start = time.time()
    # extract props
    master_df = extract_data(img_paths=img_paths,
                             exp_path=img_folder.parent,
                             masks_fold=['Masks_GNN_Track'],
                             ref_masks_fold=['Masks_laser'],
                             pixel_resolution=0.649843843874274,
                             num_chunks=2,
                             overwrite=True)
    end = time.time()
    print(f"Processing time: {end-start}")
    



# def masks_process_dict(masks_class: LoadClass)-> dict:
#     masks_dict ={}
#     for field in fields(masks_class):
#         name = field.name
#         channels = list(getattr(masks_class,name).keys())
#         if channels:
#             masks_dict[name] = channels
#     return masks_dict

# def trim_masks_list(masks_dict: dict)-> dict:
#     if 'iou_tracking' in masks_dict:
#         del masks_dict['cellpose_seg']
#     return masks_dict

# def gen_input_data(exp_obj: Experiment, img_folder_src: PathLike)-> list[PathLike]:
#     masks_dict = exp_obj.analyzed_channels
#     masks_dict = trim_masks_list(masks_dict)
    
#     img_path_list = img_list_src(exp_obj,img_folder_src)
#     img_path_input = [[file for file in img_path_list if file.__contains__(f"_f{(i+1):04d}")] for i in range(exp_obj.img_properties.n_frames)]
    
#     mask_input_list = []
#     for mask_name,mask_channels in masks_dict.items():
#         mask_path_list = mask_list_src(exp_obj,mask_name)
#         mask_keys = get_mask_keys(mask_name,exp_obj)
#         for chan in mask_channels:
#             mask_path_list = [file for file in mask_path_list if chan in file]
#             mask_path_list_per_frame = [[file for file in mask_path_list if f"_f{(i+1):04d}" in file] for i in range(exp_obj.img_properties.n_frames)]
#             for i in range(len(mask_path_list_per_frame)):
#                 mask_input_list.append({'mask_list':mask_path_list_per_frame[i],'img_list':img_path_input[i],
#                     'mask_chan':chan,'mask_name':mask_name,'exp_obj':exp_obj,'mask_keys':mask_keys})
#     return mask_input_list
   
# def get_mask_keys(mask_name: str, exp_obj: Experiment)-> list:
#     default_keys = ['cell','frames','time','mask_name','mask_chan','exp']  
    
#     if mask_name == 'iou_tracking':
#         specific_keys = exp_obj.active_channel_list
#         return default_keys + specific_keys
#     else:
#         specific_keys = exp_obj.active_channel_list
#         return default_keys + specific_keys

# def change_df_dtype(df: pd.DataFrame, exp_obj: Experiment)-> pd.DataFrame:
#     dtype_default = {'cell':'string','frames':'int','time':'float','mask_name':'category','mask_chan':'category',
#                   'exp':'category','level_0_tag':'category','level_1_tag':'category'}
#     dtype_channels = {chan:'float' for chan in exp_obj.active_channel_list}
    
#     dtype_final = {**dtype_default,**dtype_channels}
#     df = df.astype(dtype_final)
#     return df

# def extract_mask_data(mask_input_dict: dict)-> pd.DataFrame:
#     df = pd.DataFrame()
#     for _,input_dict in mask_input_dict.items():
#         frame = int(input_dict['mask_list'][0].split(sep)[-1].split('_')[2][1:])-1
#         mask = load_stack(input_dict['mask_list'],channel_list=[input_dict['mask_chan']],frame_range=[frame])
#         exp_obj = input_dict['exp_obj']
#         if mask.ndim==3:
#             mask = np.amax(mask,axis=0).astype('uint16')
#         data_dict = {k:[] for k in input_dict['mask_keys']}
#         for cell in list(np.unique(mask))[1:]:
#             data_dict['cell'].append(f"{exp_obj.exp_path.split(sep)[-1]}_{input_dict['mask_name']}_{input_dict['mask_chan']}_cell{cell}")
#             data_dict['frames'].append(frame+1)
#             data_dict['time'].append(exp_obj.time_seq[frame])
#             data_dict['mask_name'].append(input_dict['mask_name'])
#             data_dict['mask_chan'].append(input_dict['mask_chan'])
#             data_dict['exp'].append(exp_obj.exp_path.split(sep)[-1])
#             for chan in exp_obj.active_channel_list:
#                 img = load_stack(input_dict['img_list'],channel_list=[chan],frame_range=[frame])
#                 data_dict[chan].append(np.nanmean(a=img,where=mask==cell))
#         df = pd.concat([df,pd.DataFrame.from_dict(data_dict)],ignore_index=True) 
#     return df  
    
# def extract_mask_data_para(input_dict: list)-> dict:
#     frame = int(input_dict['mask_list'][0].split(sep)[-1].split('_')[2][1:])-1
#     mask = load_stack(input_dict['mask_list'],channel_list=[input_dict['mask_chan']],frame_range=[frame])
#     if mask.ndim==3:
#         mask = np.amax(mask,axis=0).astype('uint16')
#     data_dict = {k:[] for k in input_dict['mask_keys']}
#     for cell in list(np.unique(mask))[1:]:
#         data_dict['cell'].append(f"{input_dict['exp_obj'].exp_path.split(sep)[-1]}_{input_dict['mask_name']}_{input_dict['mask_chan']}_cell{cell:03d}")
#         data_dict['frames'].append(frame+1)
#         data_dict['time'].append(input_dict['exp_obj'].time_seq[frame])
#         data_dict['mask_name'].append(input_dict['mask_name'])
#         data_dict['mask_chan'].append(input_dict['mask_chan'])
#         data_dict['exp'].append(input_dict['exp_obj'].exp_path.split(sep)[-1])
#         for chan in input_dict['exp_obj'].active_channel_list:
#             img = load_stack(input_dict['img_list'],channel_list=[chan],frame_range=[frame])
#             data_dict[chan].append(np.nanmean(a=img,where=mask==cell))
#     return data_dict

# # # # # # # # # main functions # # # # # # # # # 
# def extract_channel_data(exp_obj_lst: list[Experiment], img_folder_src: PathLike, 
#                          data_overwrite: bool=False)-> list[Experiment]:
    
#     for exp_obj in exp_obj_lst:
#         # Load df
#         df_analysis = exp_obj.load_df_analysis(data_overwrite)
#         if not df_analysis.empty:
#             print(f" --> Cell data have already been extracted")
#             df_analysis = change_df_dtype(df_analysis,exp_obj) # TODO: I don't have to do this, remove later
#             continue
        
#         print(f" --> Extracting cell data")   
#         # Pre-load masks and images path
#         mask_input_list = gen_input_data(exp_obj,img_folder_src)
        
#         # Use parallel processing
#         df = pd.DataFrame()
#         with ProcessPoolExecutor() as executor:
#             data_dictS = executor.map(extract_mask_data_para,mask_input_list)
#             for data_dict in data_dictS:
#                 df = pd.concat([df,pd.DataFrame.from_dict(data_dict)],ignore_index=True)
                
#         # # Don't use parallel processing
#         # df = extract_mask_data(mask_input_list)
            
#         # Add tags
#         df['level_0_tag'] = exp_obj.analysis.level_0_tag
#         df['level_1_tag'] = exp_obj.analysis.level_1_tag
        
#         # Concat all df
#         df_analysis = pd.concat([df_analysis,df],ignore_index=True)        
#         df_analysis = change_df_dtype(df_analysis,exp_obj)
#         df_analysis = df_analysis.sort_values(by=['frames','cell'])
        
#         # Save df
#         exp_obj.save_df_analysis(df_analysis)
#         exp_obj.save_as_json()
#     return exp_obj_lst



########################### Potential functions ###########################

# def unpack_moments(df: pd.DataFrame, channels: list[str])-> pd.DataFrame:
#     """Unpack the moments from the regionprops DataFrame and add them as separate columns.
#     The original output is a list of 2D arrays containing the moments for each channel 
#     (one dim per channel). This function will unpack the array dimensions, repack it as 
#     a list of 1D arrays containing the moments for each channel, which will be reinserted as separate columns.
    
#     Args:
#         df (pd.DataFrame): DataFrame containing the regionprops output.
#         channels (list[str]): List of channel names.
#     Returns:
#         pd.DataFrame: DataFrame with the unpacked moments as separate columns for each channel."""
    
#     attributes = ['moments_weighted','moments_weighted_central','moments_weighted_normalized','moments_weighted_hu']
#     for attr in attributes:
#         data = {channel:[] for channel in channels}
#         for moment in df[attr]:
#             for i,channel in enumerate(channels):
#                 if attr == 'moments_weighted_hu':
#                     data[channel].append([array[i] for array in moment])
#                 else:
#                     data[channel].append([array[:,i] for array in moment])
        
#         attr_ind = df.columns.get_loc(attr)
#         for i, channel in enumerate(channels):
#             if i==0:
#                 df[attr] = data[channel]
#                 df.rename(columns={attr: f'{attr}_{channel}'}, inplace=True)
#             else:
#                 df.insert(attr_ind+i, f'{attr}_{channel}', data[channel])
#     return df

# def unpack_intensity(df: pd.DataFrame, channels: list[str])-> pd.DataFrame:
#     """Unpack the intensity values from the regionprops DataFrame and add them as separate columns.
#     The original output a tuple of intensities for each channel. This function will simply unpack
#     the tuple and add the intensity values as separate columns for each channel.
    
#     Args:
#         df (pd.DataFrame): DataFrame containing the regionprops output.
#         channels (list[str]): List of channel names.
#     Returns:
#         pd.DataFrame: DataFrame with the unpacked intensity values as separate columns for each channel."""
    
#     attributes = ['intensity_max','intensity_mean', 'intensity_min']
#     for attr in attributes:
#         data = {channels[i]:[intensity[i] for intensity in df[attr]] for i in range(len(channels))}
#         attr_ind = df.columns.get_loc(attr)
#         for i, channel in enumerate(channels):
#             if i==0:
#                 df[attr] = data[channel]
#                 df.rename(columns={attr: f'{attr}_{channel}'}, inplace=True)
#             else:
#                 df.insert(attr_ind+i, f'{attr}_{channel}', data[channel])
#     return df

# def unpack_centroids(df: pd.DataFrame, channels: list[str])-> pd.DataFrame:
#     """Unpack the centroids from the regionprops DataFrame and add them as separate columns.
#     The original output is a list of 2 arrays, where the first array is the y-coordinates and 
#     the second array is the x-coordinates of all the channels. This function will convert those
#     arrays into tuple pairs of (y,x) and add them as separate columns for each channel.
    
#     Args:
#         df (pd.DataFrame): DataFrame containing the regionprops output.
#         channels (list[str]): List of channel names.
#     Returns:
#         pd.DataFrame: DataFrame with the unpacked centroids as separate columns for each channel.""" 
    
#     attributes = ['centroid_weighted', 'centroid_weighted_local']
    
#     for attr in attributes:
#         attr_values = [list(zip(*df[attr][i])) for i in range(len(df[attr]))]
#         data = {channels[i]: [centroid[i] for centroid in attr_values] for i in range(len(channels))}
#         attr_ind = df.columns.get_loc(attr)
#         for i, channel in enumerate(channels):
#             if i==0:
#                 df[attr] = data[channel]
#                 df.rename(columns={attr: f'{attr}_{channel}'}, inplace=True)
#             else:
#                 df.insert(attr_ind+i, f'{attr}_{channel}', data[channel])
#     return df

# ##################### Original functions #####################
# # https://github.com/chigozienri/regionprops_to_df
# def scalar_attributes_list(im_props):
#     """
#     Makes list of all scalar, non-dunder, non-hidden
#     attributes of skimage.measure.regionprops object
#     """
    
#     attributes_list = []
    
#     for attribute in dir(im_props[0]):
#         if attribute[:1] == '_':
#             continue
#         if 'image' in attribute:
#             continue
#         attributes_list += [attribute]
            
#     return attributes_list

# def regionprops_to_df(im_props):
#     """
#     Read content of all attributes for every item in a list
#     output by skimage.measure.regionprops
#     """

#     attributes_list = scalar_attributes_list(im_props)

#     # Initialise list of lists for parsed data
#     parsed_data = []

#     # Put data from im_props into list of lists
#     for i, _ in enumerate(im_props):
#         parsed_data += [[]]
        
#         for j in range(len(attributes_list)):
#             parsed_data[i] += [getattr(im_props[i], attributes_list[j])]

#     # Return as a Pandas DataFrame
#     return pd.DataFrame(parsed_data, columns=attributes_list)

