from __future__ import annotations
from dataclasses import dataclass
from os import PathLike
from typing import Any
from pipeline.image_handeling.Base_Module_Class import BaseModule
from pipeline.image_handeling.Experiment_Classes import Experiment, init_from_json
from pipeline.segmentation.cp_segmentation import cellpose_segmentation
from pipeline.segmentation.segmentation import threshold
from pipeline.settings.Setting_Classes import Settings

@dataclass
class Segmentation(BaseModule):
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
            
    def segment_from_settings(self, settings: dict)-> list[Experiment]:
        sets = Settings(settings)
        if not hasattr(sets,'segmentation'):
            print("No segmentation settings found")
            self.save_as_json()
            return self.exp_obj_lst
        sets = sets.segmentation
        
        if hasattr(sets,'cellpose'):
            self.exp_obj_lst = self.cellpose(**sets.cellpose)
        if hasattr(sets,'threshold'):
            self.exp_obj_lst = self.thresholding(**sets.threshold)
        self.save_as_json()
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
       
    def thresholding(self, channel_to_seg: str | list[str], overwrite: bool=False, manual_threshold: int=None, img_fold_src: str="")-> list[Experiment]:
        
        if isinstance(channel_to_seg,str):
            return threshold(self.exp_obj_lst,channel_to_seg,overwrite,manual_threshold,img_fold_src)
        
        if isinstance(channel_to_seg,list):
            for channel in channel_to_seg:
                self.exp_obj_lst = self.thresholding(channel,overwrite,manual_threshold,img_fold_src)
            return self.exp_obj_lst
                
                
