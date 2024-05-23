from __future__ import annotations
from os import PathLike, remove
from os.path import exists, join
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from skimage.measure import regionprops_table
from functools import partial


PROPERTIES = ['area','centroid','intensity_mean',
                       'label','perimeter','slice','solidity']

########################### Main functions ###########################
def extract_data(img_array: np.ndarray, mask_array: np.ndarray, channels: list[str], 
                 save_path: PathLike, ref_masks: dict[str,np.ndarray] = None, overwrite: bool = False)-> pd.DataFrame:
    """Extract img properties (skimage.measure.regionprops_table) with provided mask. Img and mask must have 
    the same shape, except that img can have an optional channel dimension. Frame dimension is also optional. 
    The shape of img must be ([F],Y,X,[C]) and the shape of mask must be ([F],Y,X). If channel dim is present,
    the mean intensity of each channel will be extracted. Same for frame dim. The extracted data can be
    saved to the provided path. The data will be returned as a pandas.DataFrame.
        
    Args:
        img_array ([[F],Y,X,[C]], np.ndarray): The image array to extract the mean intensities from. Frame and channel dim are optional.
        
        mask_array ([[F],Y,X], np.ndarray): The mask array to extract the mean intensities from. Frame dim is optional.
        
        channels (list[str]): List of channel names.
        
        save_path (PathLike): The path to save the extracted data (as csv) with the name provided.
        
        ref_masks (dict[str,tuple[np.ndarray,float]], optional): Dictionary of reference masks to extract the dmap from. With keys as the reference 
        label names and values as tuple of the reference array and image resolution. Defaults to None. 
        
        overwrite (bool, optional): If True, the data will be extracted and overwrite previous extraction. Defaults to False.
    
    Returns:
        pd.DataFrame: The extracted data."""
    
    # Check if the data has already been extracted
    save_path = join(save_path, "regionprops.csv")
    if exists(save_path) and not overwrite:
        print(f"Data has already been extracted. Loading data from {save_path}")
        return pd.read_csv(save_path)
    
    # Remove the previous data if overwrite is True
    if exists(save_path):
        remove(save_path)
    
    # Prepare the renaming of the columns
    col_rename = rename_columns(img_array.ndim, mask_array.ndim, channels)
    
    # Log
    print(f"Extracting data from {save_path}")
    
    # If the mask_array is not a time sequence
    if mask_array.ndim ==2:
        prop = regionprops_table(mask_array,intensity_image=img_array,separator='_',
                                 properties=PROPERTIES,extra_properties=[dmap])
        master_df = get_regionprops(0,mask_array,img_array,ref_masks)
        master_df.rename(columns=col_rename, inplace=True)
        master_df.to_csv(save_path, index=False)
        return master_df
    
    # If the mask_array is a time sequence
    get_regionprops_partial = partial(get_regionprops, mask_array=mask_array, img_array=img_array, ref_masks=ref_masks)
    with ThreadPoolExecutor() as executor:
        lst_df = executor.map(get_regionprops_partial, range(mask_array.shape[0]))
    
    # Create the DataFrame and save to csv
    master_df = pd.concat(lst_df, ignore_index=True)
    master_df.rename(columns=col_rename, inplace=True)
    master_df.to_csv(save_path, index=False)
    return master_df

def get_regionprops(frame_idx: int, mask_array: np.ndarray, img_array: np.ndarray, ref_masks: dict[str,dict])-> pd.DataFrame:
        """Function to extract the regionprops from the mask_array and img_array. The function will extract the
        properties defined in the PROPERTIES list. If the ref_masks dictionary is provided, the function will extract
        the dmap from the reference masks. The extracted data will be returned as a pandas.DataFrame.
        
        Args:
            frame (int): The frame index to extract the data from.
            
            mask_array ([[F],Y,X], np.ndarray): The mask array to extract the regionprops from. Frame dim is optional.
            
            img_array ([[F],Y,X,[C]], np.ndarray): The image array to extract the mean intensities from. Frame and channel dim are optional.
            
            ref_masks (dict[str,tuple[np.ndarray,float]]): Dictionary of reference masks to extract the dmap from. With keys as the reference 
            label names and values as tuple of the reference array and image resolution.
            
        Returns:
            pd.DataFrame: The extracted regionprops data."""
            
            
        # Extract the regionprops
        if mask_array.ndim == 2:
            prop = regionprops_table(mask_array,intensity_image=img_array,
                                 properties=PROPERTIES, separator='_')
        else:
            prop = regionprops_table(mask_array[frame_idx],intensity_image=img_array[frame_idx],
                                 properties=PROPERTIES, separator='_')
        
        # Extract the dmap from the reference masks
        if ref_masks:
            for ref_name,ref_tuple in ref_masks.items():
                ref_mask,resolution = ref_tuple
                # Check if the experiment is a time sequence
                if mask_array.ndim == 2:
                    prop_ref = regionprops_table(mask_array, intensity_image=ref_mask,
                                             properties=['label'], separator='_',extra_properties=[dmap])
                else:
                    prop_ref = regionprops_table(mask_array[frame_idx], intensity_image=ref_mask[frame_idx],
                                             properties=['label'], separator='_',extra_properties=[dmap])
                prop[f'dmap_um_{ref_name}'] = prop_ref['dmap']*resolution
        df = pd.DataFrame(prop)
        df['frame'] = frame_idx+1
        return df

def rename_columns(img_dim: int, mask_dim: int, channels: list[str])-> dict[str,str]:
    """Function to rename the columns of the regionprops_table output. The columns will be renamed
    with the channels names.
    
    Args:
        img_dim (int): The dimension of the img_array.
        
        mask_dim (int): The dimension of the mask_array.
        
        channels (list[str]): List of channel names.
        
    Returns:
        dict[str,str]: Dictionary of the columns to rename with old name as key and new name as value."""
    
    col_rename = {'centroid_0':'centroid_y','centroid_1':'centroid_x'}
    # If the img_array has a channel dimension
    if img_dim != mask_dim: 
        col_rename = {**col_rename, **{f'intensity_mean_{i}': f'intensity_mean_{channels[i]}' for i in range(len(channels))}}
    else:
        col_rename['intensity_mean'] = f'intensity_mean_{channels[0]}'
        col_rename['intensity_centroid'] = f'intensity_centroid_{channels[0]}'
    return col_rename

def dmap(mask_region: np.ndarray, intensity_image: np.ndarray)-> int:
    """Function that extract the intensity at the centroid position of the mask region. 
    This function is used as an extra property in the regionprops_table function to extract
    the value of the dmap at the centroid position of the mask region.
    
    Args:
        mask_region (np.ndarray): The mask region to extract the intensity from.
        
        intensity_image (np.ndarray): The image array to extract the intensity from.
        
    Returns:
        int: The intensity value at the centroid position of the mask region."""
        
        
    # Calculate the centroid coordinates as integers of mask
    y, x = np.nonzero(mask_region)
    y, x = int(np.mean(y)), int(np.mean(x))
    # Return the intensity at the centroid position
    return int(intensity_image[y,x])


# # # # # # # # # Test
if __name__ == "__main__":
    import time
    from os import listdir
    from os.path import join 
    from tifffile import imread, imwrite
    from pipeline.image_handeling.data_utility import load_stack
    from scipy.ndimage import distance_transform_edt
    
    
    start = time.time()
    img_path = "/home/New_test/control/c2z25t23v1_s1/Images_Registered"
    img_files = [join(img_path,file) for file in sorted(listdir(img_path))]
    img = load_stack(img_files,["GFP","RFP"],range(23),True)
    
    mask_path = "/home/New_test/control/c2z25t23v1_s1/Masks_IoU_Track"
    mask_files = [join(mask_path,file) for file in sorted(listdir(mask_path))]
    mask = load_stack(mask_files ,["RFP"],range(23),True)
    
    save_path = '/home/New_test/test'
    ref_path = '/home/New_test/control/c2z25t23v1_s1/Masks_wound'
    ref_files = [join(ref_path,file) for file in sorted(listdir(ref_path))]
    ref = load_stack(ref_files ,["RFP"],range(23),True)
    ref = distance_transform_edt(np.logical_not(ref))
    imwrite(join(save_path,'ref.tif'),ref.astype('uint16'))
    ref_dict = {'wound':(ref,0.322),'NTC':(ref,0.322)}
    
    # img = imread(img_path)
    # stack = imread(stack_path)
    # stack = np.moveaxis(stack, [1], [-1])
    # mask = imread(mask_path)
    # ref_mask = {'wound':(imread(mask_ref)[0],0.322),'NTC':(imread(mask_ref)[0],0.322)}
    
    master_df = extract_data(img, mask, ['RFP','GFP'], save_path,ref_masks=ref_dict,overwrite=True)
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

