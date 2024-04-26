from __future__ import annotations
from os import sep, PathLike
from dataclasses import fields
import pandas as pd
import numpy as np
from image_handeling.Experiment_Classes import Experiment, LoadClass
from image_handeling.data_utility import load_stack, img_list_src, mask_list_src
from concurrent.futures import ProcessPoolExecutor


def masks_process_dict(masks_class: LoadClass)-> dict:
    masks_dict ={}
    for field in fields(masks_class):
        name = field.name
        channels = list(getattr(masks_class,name).keys())
        if channels:
            masks_dict[name] = channels
    return masks_dict

def trim_masks_list(masks_dict: dict)-> dict:
    if 'iou_tracking' in masks_dict:
        del masks_dict['cellpose_seg']
    return masks_dict

def gen_input_data(exp_obj: Experiment, img_folder_src: PathLike)-> list[PathLike]:
    masks_dict = exp_obj.analyzed_channels
    masks_dict = trim_masks_list(masks_dict)
    
    _, img_path_list = img_list_src(exp_obj,img_folder_src)
    img_path_input = [[file for file in img_path_list if file.__contains__(f"_f{(i+1):04d}")] for i in range(exp_obj.img_properties.n_frames)]
    
    mask_input_list = []
    for mask_name,mask_channels in masks_dict.items():
        mask_path_list = mask_list_src(exp_obj,mask_name)
        mask_keys = get_mask_keys(mask_name,exp_obj)
        for chan in mask_channels:
            mask_path_list = [file for file in mask_path_list if chan in file]
            mask_path_list_per_frame = [[file for file in mask_path_list if f"_f{(i+1):04d}" in file] for i in range(exp_obj.img_properties.n_frames)]
            for i in range(len(mask_path_list_per_frame)):
                mask_input_list.append({'mask_list':mask_path_list_per_frame[i],'img_list':img_path_input[i],
                    'mask_chan':chan,'mask_name':mask_name,'exp_obj':exp_obj,'mask_keys':mask_keys})
    return mask_input_list
   
def get_mask_keys(mask_name: str, exp_obj: Experiment)-> list:
    default_keys = ['cell','frames','time','mask_name','mask_chan','exp']  
    
    if mask_name == 'iou_tracking':
        specific_keys = exp_obj.active_channel_list
        return default_keys + specific_keys
    else:
        specific_keys = exp_obj.active_channel_list
        return default_keys + specific_keys

def change_df_dtype(df: pd.DataFrame, exp_obj: Experiment)-> pd.DataFrame:
    dtype_default = {'cell':'string','frames':'int','time':'float','mask_name':'category','mask_chan':'category',
                  'exp':'category','level_0_tag':'category','level_1_tag':'category'}
    dtype_channels = {chan:'float' for chan in exp_obj.active_channel_list}
    
    dtype_final = {**dtype_default,**dtype_channels}
    df = df.astype(dtype_final)
    return df

def extract_mask_data(mask_input_dict: dict)-> pd.DataFrame:
    df = pd.DataFrame()
    for _,input_dict in mask_input_dict.items():
        frame = int(input_dict['mask_list'][0].split(sep)[-1].split('_')[2][1:])-1
        mask = load_stack(input_dict['mask_list'],channel_list=[input_dict['mask_chan']],frame_range=[frame])
        exp_obj = input_dict['exp_obj']
        if mask.ndim==3:
            mask = np.amax(mask,axis=0).astype('uint16')
        data_dict = {k:[] for k in input_dict['mask_keys']}
        for cell in list(np.unique(mask))[1:]:
            data_dict['cell'].append(f"{exp_obj.exp_path.split(sep)[-1]}_{input_dict['mask_name']}_{input_dict['mask_chan']}_cell{cell}")
            data_dict['frames'].append(frame+1)
            data_dict['time'].append(exp_obj.time_seq[frame])
            data_dict['mask_name'].append(input_dict['mask_name'])
            data_dict['mask_chan'].append(input_dict['mask_chan'])
            data_dict['exp'].append(exp_obj.exp_path.split(sep)[-1])
            for chan in exp_obj.active_channel_list:
                img = load_stack(input_dict['img_list'],channel_list=[chan],frame_range=[frame])
                data_dict[chan].append(np.nanmean(a=img,where=mask==cell))
        df = pd.concat([df,pd.DataFrame.from_dict(data_dict)],ignore_index=True) 
    return df  
    
def extract_mask_data_para(input_dict: list)-> dict:
    frame = int(input_dict['mask_list'][0].split(sep)[-1].split('_')[2][1:])-1
    mask = load_stack(input_dict['mask_list'],channel_list=[input_dict['mask_chan']],frame_range=[frame])
    if mask.ndim==3:
        mask = np.amax(mask,axis=0).astype('uint16')
    data_dict = {k:[] for k in input_dict['mask_keys']}
    for cell in list(np.unique(mask))[1:]:
        data_dict['cell'].append(f"{input_dict['exp_obj'].exp_path.split(sep)[-1]}_{input_dict['mask_name']}_{input_dict['mask_chan']}_cell{cell:03d}")
        data_dict['frames'].append(frame+1)
        data_dict['time'].append(input_dict['exp_obj'].time_seq[frame])
        data_dict['mask_name'].append(input_dict['mask_name'])
        data_dict['mask_chan'].append(input_dict['mask_chan'])
        data_dict['exp'].append(input_dict['exp_obj'].exp_path.split(sep)[-1])
        for chan in input_dict['exp_obj'].active_channel_list:
            img = load_stack(input_dict['img_list'],channel_list=[chan],frame_range=[frame])
            data_dict[chan].append(np.nanmean(a=img,where=mask==cell))
    return data_dict

# # # # # # # # main functions # # # # # # # # # 
def extract_channel_data(exp_obj_lst: list[Experiment], img_folder_src: PathLike, 
                         data_overwrite: bool=False)-> list[Experiment]:
    
    for exp_obj in exp_obj_lst:
        # Load df
        df_analysis = exp_obj.load_df_analysis(data_overwrite)
        if not df_analysis.empty:
            print(f" --> Cell data have already been extracted")
            df_analysis = change_df_dtype(df_analysis,exp_obj) # TODO: I don't have to do this, remove later
            continue
        
        print(f" --> Extracting cell data")   
        # Pre-load masks and images path
        mask_input_list = gen_input_data(exp_obj,img_folder_src)
        
        # Use parallel processing
        df = pd.DataFrame()
        with ProcessPoolExecutor() as executor:
            data_dictS = executor.map(extract_mask_data_para,mask_input_list)
            for data_dict in data_dictS:
                df = pd.concat([df,pd.DataFrame.from_dict(data_dict)],ignore_index=True)
                
        # # Don't use parallel processing
        # df = extract_mask_data(mask_input_list)
            
        # Add tags
        df['level_0_tag'] = exp_obj.analysis.level_0_tag
        df['level_1_tag'] = exp_obj.analysis.level_1_tag
        
        # Concat all df
        df_analysis = pd.concat([df_analysis,df],ignore_index=True)        
        df_analysis = change_df_dtype(df_analysis,exp_obj)
        df_analysis = df_analysis.sort_values(by=['frames','cell'])
        
        # Save df
        exp_obj.save_df_analysis(df_analysis)
        exp_obj.save_as_json()
    return exp_obj_lst

