from __future__ import annotations
from dataclasses import dataclass
from os import PathLike
from pipeline.image_handeling.Base_Module_Class import BaseModule
from pipeline.image_handeling.Experiment_Classes import Experiment, init_from_json
from pipeline.tracking.iou_tracking import iou_tracking
from pipeline.settings.Setting_Classes import Settings

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
        sets = Settings(settings)
        if not hasattr(sets,'tracking'):
            print("No tracking settings found")
            self.save_as_json()
            return self.exp_obj_lst
        sets = sets.tracking

        if hasattr(sets,'iou_track'):
            self.exp_obj_lst = self.iou_tracking(**sets.iou_track)
        self.save_as_json()
        return self.exp_obj_lst
    
    def iou_tracking(self, channel_to_track: str | list[str], img_fold_src: PathLike = "", stitch_thres_percent: float=0.75, shape_thres_percent: float=0.2,
                 overwrite: bool=False, mask_appear: int=5, copy_first_to_start: bool=True, copy_last_to_end: bool=True)-> list[Experiment]:
        if isinstance(channel_to_track, str):
            return iou_tracking(self.exp_obj_lst,channel_to_track,img_fold_src,stitch_thres_percent,shape_thres_percent,overwrite,mask_appear,copy_first_to_start,copy_last_to_end)
        
        if isinstance(channel_to_track,list):
            for channel in channel_to_track:
                self.exp_obj_lst = self.iou_tracking(channel,img_fold_src,stitch_thres_percent,shape_thres_percent,overwrite,mask_appear,copy_first_to_start,copy_last_to_end)
            return self.exp_obj_lst