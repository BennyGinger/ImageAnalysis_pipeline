from __future__ import annotations
from dataclasses import dataclass
from os import PathLike
from pipeline.image_handeling.Base_Module_Class import BaseModule
from pipeline.image_handeling.Experiment_Classes import Experiment, init_from_json
from pipeline.tracking.iou_tracking import iou_tracking
from pipeline.tracking.gnn_tracking import gnn_tracking
from pipeline.tracking.man_tracking import man_tracking
from pipeline.settings.Setting_Classes import Settings

@dataclass
class TrackingModule(BaseModule):
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
    
        if hasattr(sets,'gnn_track'):
            self.exp_obj_lst = self.gnn_tracking(**sets.gnn_track)
            self.save_as_json()
            return self.exp_obj_lst
        
        if hasattr(sets,'man_track'):
            self.exp_obj_lst = self.man_tracking(**sets.man_track)
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
        
    def gnn_tracking(self, channel_to_track: str | list[str], img_fold_src: PathLike ="", model: str='neutrophil', mask_fold_src: PathLike ="", morph: bool=False,
                     min_cell_size: int=20, decision_threshold: float=0.5, mask_appear: int=2, manual_correct: bool=False, overwrite: bool=False) -> list[Experiment]:
        if isinstance(channel_to_track, str):
            return gnn_tracking(self.exp_obj_lst,channel_to_track,model,overwrite,img_fold_src,mask_fold_src,morph,mask_appear,min_cell_size,decision_threshold, manual_correct)
        
        if isinstance(channel_to_track,list):
            for channel in channel_to_track:
                self.exp_obj_lst = self.gnn_tracking(channel,model,overwrite,img_fold_src,mask_fold_src,morph,mask_appear,min_cell_size,decision_threshold, manual_correct)
            return self.exp_obj_lst
        
    def man_tracking(self, channel_to_track: str | list[str], track_seg_mask: bool = False, mask_fold_src: PathLike = None,
                     csv_name: str = None, radius: int=5, copy_first_to_start: bool=True, copy_last_to_end: bool=True, mask_appear=2, 
                     dilate_value: int = 20, process_as_2D: bool=True, overwrite: bool=False) -> list[Experiment]:
        if isinstance(channel_to_track, str):
            return man_tracking(self.exp_obj_lst,channel_to_track,track_seg_mask,mask_fold_src,csv_name,radius,copy_first_to_start, copy_last_to_end, mask_appear, dilate_value, process_as_2D, overwrite)
        
        if isinstance(channel_to_track,list):
            for channel in channel_to_track:
                self.exp_obj_lst = self.man_tracking(channel,track_seg_mask,mask_fold_src,csv_name,radius,copy_first_to_start, copy_last_to_end, mask_appear, dilate_value, process_as_2D, overwrite)
            return self.exp_obj_lst