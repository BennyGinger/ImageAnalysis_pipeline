from __future__ import annotations
from dataclasses import dataclass
from os import PathLike, sep, remove
from os.path import join, exists
import numpy as np
import pandas as pd
# from concurrent.futures import ProcessPoolExecutor
# from functools import partial
from pipeline.image_handeling.Base_Module_Class import BaseModule
from pipeline.image_handeling.Experiment_Classes import Experiment, init_from_json
from pipeline.image_handeling.data_utility import load_stack, img_list_src, seg_mask_lst_src, track_mask_lst_src
from pipeline.analysis.channel_data import extract_data
from pipeline.settings.Setting_Classes import Settings

TRACKING_MASKS = ['iou_tracking','manual_tracking','gnn_tracking']
SEGMENTATION_MASKS = ['cellpose_seg','threshold_seg']
REFERENCE_MASKS = [] # TODO: Add reference masks

@dataclass
class Analysis(BaseModule):
    # Attributes from the BaseModule class:
    # input_folder: PathLike | list[PathLike]
    # exp_obj_lst: list[Experiment] = field(init=False)
    def __post_init__(self)-> None:
        super().__post_init__()
        if self.exp_obj_lst:
            return
        
        # Initialize the experiment list
        jsons_path = self.gather_all_json_path()
        self.exp_obj_lst = [init_from_json(json_path) for json_path in jsons_path]
    
    def analyze_from_settings(self, settings: dict)-> pd.DataFrame:
        # Analyze the data based on the settings
        sets = Settings(settings)
        if not hasattr(sets,'analysis'):
            print("No analysis settings found")
            return pd.DataFrame()
        sets = sets.analysis
        if hasattr(sets,'extract_data'):
            master_df = self.create_master_df(**sets.extract_data)
        self.save_as_json()
        return master_df
        
    def create_master_df(self, img_fold_src: PathLike = "", overwrite: bool=False)-> pd.DataFrame:
        all_dfs = []
        for exp_obj in self.exp_obj_lst:
            # extract the data
            all_dfs.append(self.extract_data(exp_obj, img_fold_src, overwrite))
        master_df = pd.concat(all_dfs)
        # Save the data
        save_path = join(self.input_folder,'master_df.csv')
        if exists(save_path):
            remove(save_path)
        master_df.to_csv(save_path,index=False)
        return master_df
    
    # def create_master_df(self, img_fold_src: PathLike = "", overwrite: bool=False)-> pd.DataFrame:
    #     parallel_func = partial(self.extract_data, img_fold_src=img_fold_src, overwrite=overwrite)
        
    #     with ProcessPoolExecutor() as executor:
    #         all_dfs = executor.map(parallel_func, self.exp_obj_lst)
        
    #     master_df = pd.concat(all_dfs)
    #     # Save the data
    #     save_path = join(self.input_folder,'master_df.csv')
    #     if exists(save_path):
    #         remove(save_path)
    #     master_df.to_csv(save_path,index=False)
    #     return master_df
    
    def extract_data(self, exp_obj: Experiment, img_fold_src: PathLike = "", overwrite: bool=False)-> pd.DataFrame:
        # Gather the images and masks
        img_fold_src,img_array = _load_img(exp_obj, img_fold_src)
        masks_arrays = _load_mask(exp_obj)
        # If no masks were found, then skip
        if not masks_arrays:
            print(f"No masks were found for {exp_obj.exp_path}")
            return pd.DataFrame()
        
        # Extract the data
        dfs = []
        for mask_name, mask_array in masks_arrays.items():
            df = extract_data(img_array,mask_array,channels=exp_obj.active_channel_list,
                                save_path=exp_obj.exp_path,overwrite=overwrite)
            # Add the mask name, time in seconds and experiment name
            df['mask_name'] = mask_name
            if exp_obj.analysis.interval_sec == None:
                df['time_sec'] = 0
            else: 
                df['time_sec'] = df['frame']*exp_obj.analysis.interval_sec
            df['exp_name'] = exp_obj.exp_path.rsplit(sep,1)[1]
            # Add the labels
            for i,label in enumerate(exp_obj.analysis.labels):
                df[f'tag_level_{i}'] = label
            dfs.append(df)
        
        # Save the data
        exp_df = pd.concat(dfs)
        exp_obj.analysis.analysis_type.update({'extract_data':{'img_fold_src':img_fold_src}})
        exp_obj.save_as_json()
        return exp_df

def _load_img(exp_obj: Experiment, img_fold_src: PathLike,)-> tuple[str,np.ndarray]:
    fold_src, img_files = img_list_src(exp_obj, img_fold_src)
    frames = exp_obj.img_properties.n_frames
    channels = exp_obj.active_channel_list
    return fold_src,load_stack(img_files,channels,range(frames),True)

def _get_all_masks_files(exp_obj: Experiment)-> dict[str,dict]:
    # If not a time sequence, then load segmentation masks
    if exp_obj.img_properties.n_frames == 1:
        mask_files = exp_obj.segmentation.processed_masks
        # if empty, then return
        if not mask_files:
            return {}
        # Get mask paths
        for mask_type in mask_files.keys():
            _, mask_paths = seg_mask_lst_src(exp_obj,mask_type)
            mask_files[mask_type]['mask_paths'] = mask_paths
        return mask_files
    
    # If a time sequence, then load tracking masks
    mask_files = exp_obj.tracking.processed_masks
    # if empty, then return
    if not mask_files:
        return {}
    # Get mask paths
    for mask_type in mask_files.keys():
        mask_files[mask_type]['mask_paths'] = track_mask_lst_src(exp_obj,mask_type)
    return mask_files
    
def _load_mask(exp_obj: Experiment)-> dict[str,np.ndarray]:
    # Gather masks from all mask sources
    mask_files = _get_all_masks_files(exp_obj)
    # if empty, then no masks were found
    if not mask_files:
        return {}
    # Load masks arrays
    frames = exp_obj.img_properties.n_frames
    masks_arrays = {}
    for mask_type,mask_dict in mask_files.items():
        masks_arrays.update({f"{mask_type}_{channel}": load_stack(mask_dict['mask_paths'],channel,range(frames),True)
                 for channel in mask_dict['channels']})
    return masks_arrays