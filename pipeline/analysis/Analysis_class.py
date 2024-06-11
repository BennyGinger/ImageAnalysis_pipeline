from __future__ import annotations
from dataclasses import dataclass
from os import PathLike, sep, remove, listdir
from os.path import join, exists
import numpy as np
import pandas as pd
from pipeline.utilities.Base_Module_Class import BaseModule
from pipeline.utilities.Experiment_Classes import Experiment
from pipeline.utilities.data_utility import load_stack, img_list_src, seg_mask_lst_src, track_mask_lst_src
from pipeline.utilities.pipeline_utility import progress_bar, pbar_desc
from pipeline.analysis.data_extraction import extract_data
from pipeline.settings.Setting_Classes import Settings
from pipeline.analysis.wound_mask import draw_wound_mask

TRACKING_MASKS = ['iou_tracking','manual_tracking','gnn_tracking']
SEGMENTATION_MASKS = ['cellpose_seg','threshold_seg']
REFERENCE_MASKS = [] #TODO: Add reference masks

@dataclass
class AnalysisModule(BaseModule):
    ## Attributes from the BaseModule class:
        # input_folder: PathLike | list[PathLike]
        # exp_obj_lst: list[Experiment] = field(init=False)
        # optimization: bool = False
    
    def analyze_from_settings(self, settings: dict)-> pd.DataFrame:
        # If optimization is set, then process only the first experiment
        self.optimization = settings['optimization']
            
        # Analyze the data based on the settings
        sets = Settings(settings)
        if not hasattr(sets,'analysis'):
            print("No analysis settings found")
            return pd.DataFrame()
        sets = sets.analysis
        if hasattr(sets,'draw_mask'):
            self.draw_wound_mask(**sets.draw_mask)
        if hasattr(sets,'extract_data'):
            master_df = self.create_master_df(**sets.extract_data)
        self.save_as_json()
        return master_df
        
    def create_master_df(self, img_fold_src: PathLike = "", overwrite: bool=False)-> pd.DataFrame:
        all_dfs = []
        
        for exp_obj in progress_bar(self.exp_obj_lst,
                            desc=pbar_desc("Experiments"),
                            colour='blue'):
            # extract the data
            all_dfs.append(self.extract_data(exp_obj, img_fold_src, overwrite))
        
        # Concatenate all the dataframes
        master_df = pd.concat(all_dfs)
        # Save the data
        save_path = join(self.input_folder,'master_df.csv')
        if exists(save_path):
            remove(save_path)
        master_df.to_csv(save_path,index=False)
        return master_df
    
    def extract_data(self, exp_obj: Experiment, img_fold_src: PathLike = "", overwrite: bool=False)-> pd.DataFrame:
        ## Gather the images
        img_fold_src,img_array = _load_img(exp_obj, img_fold_src)
        
        ## Gather the masks
        masks_arrays = _load_mask(exp_obj)
        # If no masks were found, then skip
        if masks_arrays == []:
            print(f"No masks were found for {exp_obj.exp_path}")
            return pd.DataFrame()
        # Load the reference masks, if any (i.e. None)
        ref_masks,ref_names,pix_resolution = _load_ref_masks(exp_obj)
        
        ## Extract the data
        # Setup extraction
        dfs = []
        for mask_array,mask_name,sec_arrays,sec_names in masks_arrays:
            df = extract_data(img_array= img_array,mask_array=mask_array,save_path=exp_obj.exp_path,channels=exp_obj.active_channel_list,overwrite=overwrite,mask_name=mask_name,reference_masks=ref_masks,ref_name=ref_names,pixel_resolution=pix_resolution,secondary_masks=sec_arrays,sec_names=sec_names)
            # Add the time in seconds and experiment name
            interval_sec = 1 if exp_obj.analysis.interval_sec is None else exp_obj.analysis.interval_sec
            df['time_sec'] = df['frame']*interval_sec
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

    def draw_wound_mask(self, mask_label: str | list[str], channel_show: str=None, overwrite: bool=False)-> None:
        if isinstance(mask_label, str):
            mask_label=[mask_label]
        
        for exp_obj in self.exp_obj_lst:
            # Activate branch and get imgage files
            exp_obj.analysis.is_reference_masks = True
            img_flod_src, img_files = img_list_src(exp_obj,None)
            # Get metadata
            metadata = {'um_per_pixel':exp_obj.analysis.um_per_pixel,
                        'finterval':exp_obj.analysis.interval_sec}
            # Draw
            draw_wound_mask(img_files,mask_label,channel_show,exp_obj.img_properties.n_frames,overwrite,metadata=metadata)
            # Save settings
            exp_obj.analysis.reference_masks.update({label:{'fold_src':img_flod_src,'channel_show':channel_show} 
                                                     for label in mask_label})
            exp_obj.save_as_json()


######################## Helper Functions ########################
def _load_img(exp_obj: Experiment, img_fold_src: PathLike)-> tuple[str,np.ndarray]:
    fold_src, img_files = img_list_src(exp_obj, img_fold_src)
    frames = exp_obj.img_properties.n_frames
    channels = exp_obj.active_channel_list
    return fold_src,load_stack(img_files,channels,range(frames),True)

def _get_segmentation_masks(exp_obj: Experiment)-> dict[str,dict]:
    mask_files = exp_obj.segmentation.processed_masks
    # if empty, then return
    if not mask_files:
        return {}
    # Get mask paths
    for mask_type in mask_files.keys():
        _, mask_paths = seg_mask_lst_src(exp_obj,mask_type)
        mask_files[mask_type]['mask_paths'] = mask_paths
    return mask_files

def _get_tracking_masks(exp_obj: Experiment)-> dict[str,dict]:
    mask_files = exp_obj.tracking.processed_masks
    # if empty, then return
    if not mask_files:
        return mask_files
    # Get mask paths
    for mask_type in mask_files.keys():
        fold_src = mask_files[mask_type]['fold_loc']
        print(fold_src)
        mask_files[mask_type]['mask_paths'] = track_mask_lst_src(exp_obj,fold_src)
    return mask_files

def _get_all_masks_files(exp_obj: Experiment)-> dict[str,dict]:
    # If not a time sequence, then load segmentation masks
    if exp_obj.img_properties.n_frames == 1:
        return _get_segmentation_masks(exp_obj)
    # If a time sequence, then load tracking masks
    return _get_tracking_masks(exp_obj)
        
def _load_mask(exp_obj: Experiment)-> list[tuple[np.ndarray,str,list[np.ndarray]|None,list[str]|None]]:
    # Gather masks from all mask sources
    mask_files = _get_all_masks_files(exp_obj)
    # if empty, then no masks were found
    if not mask_files:
        return []
    # Load masks arrays
    frames = exp_obj.img_properties.n_frames
    output_masks = []
    for mask_type,mask_dict in mask_files.items():
        channels = mask_dict['channels']
        mask_paths = mask_dict['mask_paths']
        # Pre-load all the mask arrays
        loaded_arrays = {channel:load_stack(mask_paths,channel,range(frames),True) for channel in channels}
        # Build the output masks tuples
        for channel in channels:
            array_name = f"{mask_type}_{channel}"
            sec_arrays = [loaded_arrays[chan] for chan in channels if chan != channel]
            if sec_arrays == []:
                sec_arrays = None
            sec_names = [f"{chan}" for chan in channels if chan != channel]
            if sec_names == []:
                sec_names = None
            output_masks.append((loaded_arrays[channel],array_name,sec_arrays,sec_names))
    return output_masks

def _load_ref_masks(exp_obj: Experiment)-> tuple[list[np.ndarray]|None,list[str]|None,float|None]:
    ## If no reference masks, then return
    if not exp_obj.analysis.is_reference_masks:
        return None,None,None
    
    ## Else, load the reference masks
    # Unpack varaibles
    ref_masks_dict = exp_obj.analysis.reference_masks
    exp_path = exp_obj.exp_path
    resolution = exp_obj.analysis.um_per_pixel[0]
    frames = exp_obj.img_properties.n_frames
    
    # Get mask paths
    mask_arrays = []; mask_labels = []
    for mask_label in ref_masks_dict.keys():
        # Unpack settings
        folder_path = join(exp_path,f"Masks_{mask_label}")
        mask_paths = [join(folder_path,file) for file in sorted(listdir(folder_path))]
        channel = ref_masks_dict[mask_label]['channel_show']
        # Load the mask array
        mask_arrays.append(load_stack(mask_paths,channel,range(frames),True))
        mask_labels.append(mask_label)
    return mask_arrays,mask_labels,resolution


if __name__== "__main__":
    
    input_folder = "/home/Test_images/nd2/Run2"
    aclass = AnalysisModule(input_folder)
    
    print(aclass.exp_obj_lst)
    # aclass.create_master_df(overwrite=True)