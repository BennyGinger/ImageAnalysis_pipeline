from __future__ import annotations
from dataclasses import dataclass
from os import PathLike, walk, sep
from os.path import join
from typing import Any
from image_handeling.Base_Module_Class import BaseModule
from image_handeling.Experiment_Classes import Experiment, init_from_json
from .cp_segmentation import cellpose_segmentation
from .segmentation import threshold
from settings.Setting_Classes import SegmentationSettings

@dataclass
class Segmentation(BaseModule):
    # Attributes from the BaseModule class:
        # input_folder: PathLike | list[PathLike]
        # experiment_list: list[Experiment] = field(init=False)
    def __post_init__(self)-> None:
        super().__post_init__()
        if self.exp_obj_lst:
            return
        
        # Initialize the experiment list
        jsons_path = gather_all_json_path(self.input_folder)
        self.exp_obj_lst = [init_from_json(json_path) for json_path in jsons_path]
            
    def segment_from_settings(self, settings: dict)-> list[Experiment]:
        sets = SegmentationSettings(settings)
        
        if hasattr(sets,'cellpose'):
            self.exp_obj_lst = self.cellpose(**sets.cellpose)
        if hasattr(sets,'threshold'):
            self.exp_obj_lst = self.thresholding(**sets.threshold)
        return self.exp_obj_lst
    
    def cellpose(self, channel_to_seg: str | list[str], model_type: str | PathLike = 'cyto3', diameter: float = 60, flow_threshold: float = 0.4, 
                 cellprob_threshold: float = 0, overwrite: bool = False, img_fold_src: str = "", process_as_2D: bool = False, save_as_npy: bool = False,
                 nuclear_marker: str = "",**kwargs: Any)-> list[Experiment]:
        
        if isinstance(channel_to_seg,str):
            return cellpose_segmentation(self.exp_obj_lst,channel_to_seg,model_type,diameter,flow_threshold,
                                         cellprob_threshold,overwrite,img_fold_src,process_as_2D,save_as_npy,nuclear_marker,**kwargs)
        if isinstance(channel_to_seg,list):
            for channel in channel_to_seg:
                self.exp_obj_lst = self.cellpose(channel,model_type,diameter,flow_threshold,cellprob_threshold,
                              overwrite,img_fold_src,process_as_2D,save_as_npy,nuclear_marker,**kwargs)
            return self.exp_obj_lst
        raise ValueError("channel_to_seg should be a string or a list of strings")
       
    def thresholding(self, channel_to_seg: str | list[str], overwrite: bool=False, manual_threshold: int=None, img_fold_src: str="")-> list[Experiment]:
        
        if isinstance(channel_to_seg,str):
            return threshold(self.exp_obj_lst,channel_to_seg,overwrite,manual_threshold,img_fold_src)
        
        if isinstance(channel_to_seg,list):
            for channel in channel_to_seg:
                self.exp_obj_lst = self.thresholding(channel,overwrite,manual_threshold,img_fold_src)
            return self.exp_obj_lst      
        raise ValueError("channel_to_seg should be a string or a list of strings")   
                
                
def gather_all_json_path(input_folder: str | list[str])-> list[str]:
    print(f"\nSearching for 'exp_settings.json' files in {input_folder}")
    # Get the path of all the json files in all subsequent folders/subfolders
    if isinstance(input_folder,str):
        return get_json_path(input_folder)
    
    if isinstance(input_folder,list):
        jsons_path = []
        for folder in input_folder:
            jsons_path.extend(get_json_path(folder))
        return jsons_path
       
def get_json_path(folder: str)-> list[str]:
    # Get the path of all the nd2 files in all subsequent folders/subfolders and exp_dict if available
    jsons_path = []
    for root , _, files in walk(folder):
        for f in files:
            # Look for all files with selected extension and that are not already processed 
            if f == 'exp_settings.json':
                jsons_path.append(join(sep,root+sep,f))
    return sorted(jsons_path)
    
    