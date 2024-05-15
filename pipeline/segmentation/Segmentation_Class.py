from __future__ import annotations
from dataclasses import dataclass
from os import PathLike
from typing import Any
from pipeline.image_handeling.Base_Module_Class import BaseModule
from pipeline.image_handeling.Experiment_Classes import Experiment, init_from_json
from pipeline.segmentation.cp_segmentation import cellpose_segmentation
from pipeline.segmentation.segmentation import threshold
from pipeline.settings.Setting_Classes import Settings
from pipeline.image_handeling.data_utility import img_list_src

@dataclass
class SegmentationModule(BaseModule):
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
            self.cellpose(**sets.cellpose)
        if hasattr(sets,'threshold'):
            self.exp_obj_lst = self.thresholding(**sets.threshold)
        self.save_as_json()
        return self.exp_obj_lst
    
    def cellpose(self, channel_to_seg: str | list[str], model_type: str | PathLike = 'cyto3', 
                 diameter: float = 60, flow_threshold: float = 0.4, 
                 cellprob_threshold: float = 0, overwrite: bool = False, 
                 img_fold_src: str = "", process_as_2D: bool = False, 
                 save_as_npy: bool = False,**kwargs: Any)-> None:
        
        if isinstance(channel_to_seg,str):
            for exp_obj in self.exp_obj_lst:
                # Activate branch
                exp_obj.segmentation.is_cellpose_seg = True
                # Get the image paths and metadata
                img_fold_src,img_paths = img_list_src(exp_obj,img_fold_src)
                metadata = {'finterval':exp_obj.analysis.interval_sec,
                            'um_per_pixel':exp_obj.analysis.um_per_pixel}
                # Run cellpose
                model_settings,cellpose_eval = cellpose_segmentation(img_paths,channel_to_seg,model_type,
                                        diameter,flow_threshold,cellprob_threshold,overwrite,process_as_2D,
                                        save_as_npy,metadata=metadata,**kwargs)
                # Save settings
                exp_obj.segmentation.cellpose_seg[channel_to_seg] = {'fold_src':img_fold_src,
                                                                     'model_settings':model_settings,
                                                                     'cellpose_eval':cellpose_eval}
                exp_obj.save_as_json()
            return
        
        if isinstance(channel_to_seg,list):
            for channel in channel_to_seg:
                self.cellpose(channel,model_type,diameter,flow_threshold,cellprob_threshold,
                              overwrite,img_fold_src,process_as_2D,save_as_npy,**kwargs)
            return
       
    def thresholding(self, channel_to_seg: str | list[str], overwrite: bool=False, manual_threshold: int=None, img_fold_src: str="")-> list[Experiment]:
        
        if isinstance(channel_to_seg,str):
            return threshold(self.exp_obj_lst,channel_to_seg,overwrite,manual_threshold,img_fold_src)
        
        if isinstance(channel_to_seg,list):
            for channel in channel_to_seg:
                self.exp_obj_lst = self.thresholding(channel,overwrite,manual_threshold,img_fold_src)
            return self.exp_obj_lst
                
                
