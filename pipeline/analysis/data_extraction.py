from __future__ import annotations
from os import PathLike, remove
from os.path import exists, join
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from skimage.measure import regionprops_table
from skimage.morphology import disk, erosion
from scipy.ndimage import distance_transform_edt
from functools import partial
from threading import Lock
from pipeline.utilities.data_utility import run_multithread, run_multiprocess

# Build-in properties for regionprops_table. NOTE: if modifying this list, make sure to update the _rename_columns() function.
PROPERTIES = ['area','centroid','intensity_mean',
                       'label','perimeter','slice','solidity']

########################### Main functions ###########################
def extract_data(img_array: np.ndarray, mask_array: np.ndarray, save_path: PathLike, channels: str | list[str]=None, overwrite: bool = False, mask_name: str=None, reference_masks: np.ndarray|list[np.ndarray]=None, ref_name: str|list[str]=None, pixel_resolution: float=None, secondary_masks: np.ndarray | list[np.ndarray]=None, sec_names: str|list[str]=None)-> pd.DataFrame:
    """Extract images properties using the skimage.measure.regionprops_table with provided mask. the image can be processed either as a time sequence (F) or as a single frame. The mask must have must have the same shape as the image. Additionally, if a channel dim (C) is provided for the image, the mean intensity of each channel will be extracted. So, the expected shapes are ([F],Y,X,[C]) for the image and ([F],Y,X) for the mask, with [F] and [C] being optional. The data will be returned as a pandas.DataFrame, which willl also be saved at the save_path provided as a csv file named 'regionprops.csv'.
    Reference masks can also be provided to extract the distance transform from the centroid of the primary mask (i.e. mask_array). The distance transform will be added to the main properties as 'dmap_um_{ref_name}' or 'dmap_pixel_{ref_name}' if the pixel resolution is not provided. The secondary masks can also be provided to check if the primary mask cells overlap with the secondary masks cells. The overlap will be added to the main properties as 'label_{sec_name}_positive'.
    
    Args:
        img_array ([[F],Y,X,[C]] np.ndarray): The image array to extract the mean intensities from. F and C dim are optional.
        
        mask_array ([[F],Y,X] np.ndarray): The mask array to extract the mean intensities from. F dim is optional, but must match the image array.
        
        save_path (PathLike): The path to save the extracted data as "regionprops.csv".
        
        channels (str | list[str], optional): channel name(s), used to rename the columns of the extracted data. If set to None, generic name(s) will be created. Defaults to None.
        
        overwrite (bool, optional): Whether to overwrite the data if it already exists. Defaults to False.
        
        mask_name (str, optional): The name of the mask. If set to None, a generic name will be given. Defaults to None.
        
        reference_masks ([[F],Y,X] np.ndarray | list[np.ndarray], optional): Reference mask(s) to extract the distance transform from the centroid of the primary mask. Defaults to None.
        
        ref_name (str | list[str], optional): The name of the reference mask(s). If set to None, generic name(s) will be given. Defaults to None.
        
        pixel_resolution (float, optional): The pixel resolution of the image to convert the distance into um. If set to None, the distance transform will be in pixel. Defaults to None.
        
        secondary_masks ([[F],Y,X] np.ndarray | list[np.ndarray], optional): Secondary mask(s) to check if the primary mask cells overlap with the secondary masks cells. Defaults to None.
        
        secondary_names (str | list[str], optional): The name of the secondary mask(s). If set to None, generic name(s) will be given. Defaults to None.
         
    Returns:
        -> pd.DataFrame"""
    
    
    # Check if the data has already been extracted
    save_path = join(save_path, "regionprops.csv")
    print(f" --> Extracting data from \033[94m{save_path}\033[0m")
    if exists(save_path) and not overwrite:
        print(f"  ---> Data has already been extracted. Loading data from \033[94m{save_path}\033[0m")
        return pd.read_csv(save_path)
    else:
        remove(save_path) if exists(save_path) else None
            
    # Setup the data extraction
    col_rename = _rename_columns(img_array.ndim, mask_array.ndim, channels)
    ref_masks = prepare_ref_masks(reference_masks,ref_name,pixel_resolution)
    sec_masks = prepare_sec_masks(secondary_masks,sec_names)
    if mask_name is None:
        mask_name = "unamed_mask"
    
    ## Extract the data
    # If the mask_array is not a time sequence
    print("  ---> Extracting the regionprops")
    if mask_array.ndim ==2:
        master_df = _extract_regionprops(None,mask_array,img_array,mask_name,ref_masks,sec_masks)
        master_df.rename(columns=col_rename, inplace=True)
        master_df.to_csv(save_path, index=False)
        return master_df
    
    # Else, if the mask_array is a time sequence
    fixed_args = {'mask_array':mask_array,'img_array':img_array,'mask_name':mask_name,'ref_masks':ref_masks,'sec_masks':sec_masks}
    lst_df = run_multithread(_extract_regionprops,range(mask_array.shape[0]),fixed_args)
    # with ThreadPoolExecutor() as executor:
    #     lst_df = executor.map(partial(_extract_regionprops,**fixed_args), range(mask_array.shape[0]))
    master_df = pd.concat(lst_df, ignore_index=True)
    master_df.rename(columns=col_rename, inplace=True)
    master_df.to_csv(save_path, index=False)
    return master_df


############################# Helper functions #############################
def prepare_ref_masks(ref_masks: np.ndarray|list[np.ndarray]=None, ref_names: str|list[str]=None, pixel_resolution: float=None)-> list[tuple[np.ndarray,str,float|None]] | None | ValueError:
    """Function to prepare the reference masks for the analysis. Dmap will be applied to the reference masks to measure the distance from the reference point/area."""
    
    
    # Retrun None if no reference masks are provided
    if ref_masks is None:
        return None

    # If no names are provided, create generic names
    if ref_names is None:
        ref_names = [f"ref{i+1}" for i in range(len(ref_masks))]
    
    # Check type of ref_masks and ref_names
    if isinstance(ref_masks, np.ndarray):
        ref_masks = [ref_masks]
    if isinstance(ref_names, str):
        ref_names = [ref_names]
    
    # Check if the number of reference masks and names match
    if len(ref_names) != len(ref_masks):
        raise ValueError(f"The number of reference masks {len(ref_masks)} must match the number of reference names {len(ref_names)}.")
    
    # If the pixel resolution is not provided, set it to None (pixel will be used)
    if pixel_resolution is None:
        pixel_resolution = [None]*len(ref_masks)
    else:
        pixel_resolution = [pixel_resolution]*len(ref_masks)
    
    # Apply the dmap to the reference masks
    print("  ---> Applying distance transform to the reference masks")
    ref_masks = run_multiprocess(_dist_transform,ref_masks)
    
    return list(zip(ref_masks,ref_names,pixel_resolution))
        
def prepare_sec_masks(secondary_masks: np.ndarray | list[np.ndarray]=None, secondary_names: str|list[str]=None)-> list[np.ndarray,str] | None | ValueError:
    """Function to prepare the secondary masks for the analysis. The secondary masks will be eroded to minimize false positive overlap between primary mask cell and secondary cells."""
    
    
    # Retrun None if no reference masks are provided
    if secondary_masks is None:
        return None
    
    # If no names are provided, create generic names
    if secondary_names is None:
        secondary_names = [f"sec{i+1}" for i in range(len(secondary_masks))]
    
    # Check type of ref_masks and ref_names
    if isinstance(secondary_masks, np.ndarray):
        secondary_masks = [secondary_masks]
    if isinstance(secondary_names, str):
        secondary_names = [secondary_names]
    
    # Check if the number of reference masks and names match
    if len(secondary_names) != len(secondary_masks):
        raise ValueError(f"The number of secondary masks {len(secondary_masks)} must match the number of secondary names {len(secondary_names)}.")
    
    # Erode secondary masks
    print("  ---> Eroding the secondary masks")
    secondary_masks = run_multiprocess(_erode_secondary_mask,secondary_masks)
    
    return list(zip(secondary_masks,secondary_names))

def _extract_regionprops(frame_idx: int | None, mask_array: np.ndarray, img_array: np.ndarray, mask_name: str, ref_masks: list[tuple[np.ndarray,str,float|None]]=None, sec_masks: list[tuple[np.ndarray,str]]=None,**kwargs)-> pd.DataFrame:
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
        df['frame'] = frame_idx+1
        df['mask_name'] = mask_name
        return df

def ref_props(ref_masks: list[np.ndarray,str,float|None], mask_array: np.ndarray, frame_idx: int, prop: dict[str,float])-> None:
    """Extract the regionprops from the reference masks. The function will compute the distance transform value from the dmap mask of the centroid of the primary mask. The distance transform value will be added to the main properties."""
    
    
    for ref_mask,ref_name,resolution in ref_masks:
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

def sec_props(sec_masks: list[tuple[np.ndarray,str]], mask_array: np.ndarray, frame_idx: int, prop: dict[str,float])-> None:
    """Extract the regionprops from the secondary masks. The function will compute the overlap between the primary mask cells and the secondary masks cells and return a boolean value, whether the primary mask cells are in the secondary masks cells."""
    
    
    for sec_mask,sec_name in sec_masks:
        if mask_array.ndim == 2:
            prop_sec = regionprops_table(mask_array,sec_mask,properties=['label'],separator='_',extra_properties=[label_in])
        else:
            prop_sec = regionprops_table(mask_array[frame_idx],sec_mask[frame_idx],properties=['label'],separator='_',extra_properties=[label_in])
            
        # Update the main properties with the overlap
        prop[f'label_classification'] = [f"{sec_name}_positive" if state else f"{sec_name}_negative" for state in prop_sec['label_in']]

def _rename_columns(img_dim: int, mask_dim: int, channels: str | list[str])-> dict[str,str]:
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
        col_rename.update({'label':'cell_ID'})
    
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

def _erode_secondary_mask(mask: np.ndarray,)-> np.ndarray:
    """Function to erode the secondary mask to minimize false positive overlap between primary mask cell and secondary cells. Mask will be eroded one cell at a time in parallel."""
    
    
    # Setup the erosion
    footprint = disk(6)
    unique_cells = np.unique(mask)[1:]
    
    # Extract the number of frames
    nframes = mask.shape[0] if mask.ndim > 2 else 1
    
    # Erode the secondary mask
    for i in range(nframes):
        with ThreadPoolExecutor() as executor:
            lock = Lock()
            eroded_frames = executor.map(partial(_erode_mask,mask=mask[i],footprint=footprint,lock=lock),unique_cells)
        mask_frame = np.zeros_like(mask[i])
        for frame in eroded_frames:
            mask_frame += frame
        mask[i] = mask_frame
    return mask

def _erode_mask(cell_idx: int, mask: np.ndarray, footprint: np.ndarray, lock: Lock)-> np.ndarray:
    """Apply the erosion to the secondary mask for a single cell."""
    
    
    with lock:
        temp_mask = np.where(mask==cell_idx,cell_idx,0)
    return erosion(temp_mask,footprint).astype('uint16')

def _dist_transform(mask: np.ndarray)-> np.ndarray:
    """Apply the distance transform to the mask."""
    
    
    return distance_transform_edt(np.logical_not(mask))    



# # # # # # # # # Test
if __name__ == "__main__":
    import time
    from os import listdir
    from os.path import join 
    from tifffile import imread, imwrite
    from pipeline.utilities.data_utility import load_stack
    from scipy.ndimage import distance_transform_edt
    
    nframes = 23
    channels = ['GFP','RFP']
    save_path = '/home/Test_images/masks'
    pixel_resolution = 0.642
    
    # Load the images stack
    img_path = "/home/Test_images/nd2/Run2/c2z25t23v1_nd2_s1/Images_Registered"
    img_files = [join(img_path,file) for file in sorted(listdir(img_path))]
    img = load_stack(img_files,channels,range(nframes),True)
   
    # RFP
    mask_name = 'RFP_IoU_Track'
    mask_path = "/home/Test_images/nd2/Run2/c2z25t23v1_nd2_s1/Masks_IoU_Track"
    mask_files = [join(mask_path,file) for file in sorted(listdir(mask_path))]
    mask = load_stack(mask_files ,channels[1],range(nframes),True)
 
    # Reference
    ref_path = '/home/Test_images/nd2/Run2/Mask.tif'
    ref = imread(ref_path)
    ref = np.stack([ref]*nframes)
    ref = [ref]
    ref_names = 'wound'
    
    # GFP
    sec_mask = load_stack(mask_files ,channels[0],range(nframes),True)
    sec_mask = [sec_mask]
    sec_names = "GFP"
    
    
    start = time.time()
    # extract props
    master_df = extract_data(img_array=img,
                             mask_array=mask,
                             save_path=save_path,
                             channels=channels,
                             mask_name=mask_name,
                             reference_masks=ref,
                             ref_name=ref_names,
                             pixel_resolution=pixel_resolution,
                             secondary_masks=sec_mask,
                             sec_names=sec_names,
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

