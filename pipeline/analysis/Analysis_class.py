from __future__ import annotations
from dataclasses import dataclass
from os import PathLike
import numpy as np
from pipeline.image_handeling.Base_Module_Class import BaseModule
from pipeline.image_handeling.Experiment_Classes import Experiment, init_from_json
from pipeline.image_handeling.data_utility import load_stack, img_list_src, seg_mask_lst_src, track_mask_lst_src
from pipeline.analysis.channel_data import extract_data
from pipeline.settings.Setting_Classes import Settings

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
    
    def analyze_from_settings(self, settings: dict)-> list[Experiment]:
        pass
    
    def extract_data(self, img_fold_src: PathLike = "", overwrite: bool=False)-> list[Experiment]:
        for exp_obj in self.exp_obj_lst:
            img_files = img_list_src(exp_obj, img_fold_src)
            frames = exp_obj.img_properties.n_frames
            channels = exp_obj.active_channel_list
            img = load_stack(img_files,channels,range(frames),True)
            
            extract_data()

def _load_img(exp_obj: Experiment, img_fold_src: PathLike,)-> tuple[np.ndarray,np.ndarray]:
    _, img_files = img_list_src(exp_obj, img_fold_src)
    frames = exp_obj.img_properties.n_frames
    channels = exp_obj.active_channel_list
    return load_stack(img_files,channels,range(frames),True)

def _list_all_masks(exp_obj: Experiment, mask_fold_src: PathLike,)-> dict:
    # Get all the masks
    all_active_masks = exp_obj.analyzed_channels
    
    # Trim the segmentation masks, if tracking exists for those masks
    tracking_names = ['iou_tracking','manual_tracking','gnn_tracking']
    if set(all_active_masks.keys()).intersection(tracking_names):
        for track_name in tracking_names:
            track_settings = getattr(exp_obj.tracking, track_name)
            
            # If tracking settings are not found, continue, meaning the tracking doesn't exist
            if not track_settings:
                continue
            
    pass

def _load_mask(exp_obj: Experiment, mask_fold_src: PathLike,)-> np.ndarray:
    pass