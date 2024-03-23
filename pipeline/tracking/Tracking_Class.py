from __future__ import annotations
from dataclasses import dataclass
from os import PathLike
from image_handeling.Base_Module_Class import BaseModule
from image_handeling.Experiment_Classes import Experiment, init_from_json
from .iou_tracking import iou_tracking
from settings.Setting_Classes import Settings

@dataclass
class Tracking(BaseModule):
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
    
    def track_from_settings(self, settings: dict)-> list[Experiment]:
        sets = Settings(settings).tracking
        
        if hasattr(sets,'iou_track'):
            self.exp_obj_lst = self.iou_tracking(**sets.iou_track)
        return self.exp_obj_lst
    
    def iou_tracking(self, channel_to_track: str | list[str], img_fold_src: PathLike = "", stitch_thres_percent: float=0.75, shape_thres_percent: float=0.2,
                 overwrite: bool=False, n_mask: int=5)-> list[Experiment]:
        if isinstance(channel_to_track, str):
            return iou_tracking(self.exp_obj_lst,channel_to_track,img_fold_src,stitch_thres_percent,shape_thres_percent,overwrite,n_mask)
        
        if isinstance(channel_to_track,list):
            for channel in channel_to_track:
                self.exp_obj_lst = self.iou_tracking(channel,img_fold_src,stitch_thres_percent,shape_thres_percent,overwrite,n_mask)
            return self.exp_obj_lst