from __future__ import annotations
from os import PathLike
from os.path import exists, join
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from skimage.measure import regionprops
from skimage.measure._regionprops import RegionProperties


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

########################### Main functions ###########################
def extract_data(img_array: np.ndarray, mask_array: np.ndarray, channels: list[str], 
                 save_path: PathLike, overwrite: bool = False)-> pd.DataFrame:
    """Extract img properties (skimage.measure.regionprops) with provided mask. Img and mask must have the same shape,
    except that img can have an optional channel dimension. Frame dimension is also optional. So the
    shape of img must be ([F],Y,X,[C]) and the shape of mask must be ([F],Y,X). If channel dim is present,
    the mean intensity of each channel will be extracted. Same for frame dim. The extracted data will be
    saved to the provided path. The data will be returned as a pd.DataFrame.
        
    Args:
        img_array ([[F],Y,X,[C]], np.ndarray): The image array to extract the mean intensities from. Frame and channel dim are optional.
        mask_array ([[F],Y,X], np.ndarray): The mask array to extract the mean intensities from. Frame dim is optional.
        channels (list[str]): List of channel names.
        save_path (PathLike): The path to save the extracted data (as csv) with the name provided.
    Returns:
        pd.DataFrame: The extracted data."""
    
    # Check if the data has already been extracted
    df_name = "regionprops.csv"
    save_path = join(save_path, df_name)
    if is_extracted(save_path) and not overwrite:
        print(f"Data has already been extracted. Loading data from {save_path}")
        return pd.read_csv(save_path)
    
    # If the mask_array is not a time sequence
    if mask_array.ndim ==2:
        prop = regionprops(mask_array, intensity_image=img_array)
        master_df = regionprops_to_df(prop)
        master_df['frame'] = 1
        master_df = unpack_intensity(master_df, channels)
        master_df.to_csv(save_path, index=False, header=True)
        return master_df
    
    # If the mask_array is a time sequence
    def get_regionprops(frame: int)-> pd.DataFrame:
        """Nested function to extract the regionprops for each frame in multi-threading."""
        prop = regionprops(mask_array[frame], intensity_image=img_array[frame])
        df = regionprops_to_df(prop)
        df['frame'] = frame+1
        return unpack_intensity(df,channels)
    
    with ThreadPoolExecutor() as executor:
        lst_df = executor.map(get_regionprops, range(mask_array.shape[0]))
    
    # Create the DataFrame and save to csv
    master_df = pd.concat(lst_df, ignore_index=True)
    master_df.to_csv(save_path, index=False, header=True)
    return master_df

########################### Helper functions ###########################
def unpack_intensity(df: pd.DataFrame, channels: str | list[str])-> pd.DataFrame:
    """Unpack the intensity values, if necessary, from the regionprops DataFrame and add them as 
    separate columns. The original output a tuple of intensities (float) for each channel. 
    This function will simply unpack the tuple and add the intensity values as separate columns 
    for each channel. If Only one channel is present, then the intensity_mean is a float, not a tuple,
    and the function will simply rename the column to intensity_mean_{channel}.
    
    Args:
        df (pd.DataFrame): DataFrame containing the regionprops output.
        channels (list[str]): List of channel names.
    Returns:
        pd.DataFrame: DataFrame with the unpacked intensity values as separate columns for each channel."""
    
    # If only one channel is present, then the intensity_mean is a float, not a tuple
    if isinstance(df.intensity_mean[0], float):
        if len(channels)!=1:
            raise ValueError("Only one channel is present in the DataFrame. Please provide only 1 channel label.")
        
        if isinstance(channels, list):
            channels = channels[0]
        
        df.rename(columns={'intensity_mean': f'intensity_mean_{channels}'}, inplace=True)
        return df
    
    # If multiple channels are present, then the intensity_mean is a tuple of floats
    if len(channels)!=len(df.intensity_mean[0]):
        raise ValueError(f"The number of channels do not match the number of intensity values. \
            Please add {len(df.intensity_mean[0])} channel labels.")
    
    data = {channels[i]:[intensity[i] for intensity in df.intensity_mean] for i in range(len(channels))}
    attr_ind = df.columns.get_loc('intensity_mean')
    for i, channel in enumerate(channels):
        if i==0:
            df.intensity_mean = data[channel]
            df.rename(columns={'intensity_mean': f"intensity_mean_{channel}"}, inplace=True)
        else:
            df.insert(attr_ind+i, f"intensity_mean_{channel}", data[channel])
    return df

def regionprops_to_df(img_props: RegionProperties)-> pd.DataFrame:
    """Read content of selected attributes for every item in a list
    output by skimage.measure.regionprops
    """

    attributes_list = ['area','centroid','intensity_mean',
                       'label','perimeter','slice','solidity']

    # Initialise list of lists for parsed data
    parsed_data = []
    # Put data from img_props into list of lists
    for i, img_prop in enumerate(img_props):
        parsed_data += [[]]
        for j in range(len(attributes_list)):
            parsed_data[i] += [getattr(img_prop, attributes_list[j])]
        
    # Return as a Pandas DataFrame
    return pd.DataFrame(parsed_data, columns=attributes_list)

def is_extracted(save_path: PathLike)-> bool:
    """Check if the data has already been extracted and saved to the provided path."""
    
    if exists(save_path):
        return True
    return False

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

# # # # # # # # # Test
if __name__ == "__main__":
    import time
    from tifffile import imread
    
    
    start = time.time()
    img_path = "/home/Test_images/masks/MAX_Images.tif"
    stack_path = "/home/Test_images/masks/MAX_Merged.tif"
    mask_path = "/home/Test_images/masks/Masks_IoU_Track.tif"

    img = imread(img_path)
    stack = imread(stack_path)
    stack = np.moveaxis(stack, [1], [-1])
    mask = imread(mask_path)
    
    master_df = extract_data(stack, mask, ['RFP','GFP'], '/home/Test_images/masks')
    end = time.time()
    print(f"Processing time: {end-start}")