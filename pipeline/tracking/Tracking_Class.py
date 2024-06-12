from __future__ import annotations
from dataclasses import dataclass
from os import PathLike
from pipeline.utilities.Base_Module_Class import BaseModule
from pipeline.utilities.Experiment_Classes import Experiment
from pipeline.utilities.data_utility import seg_mask_lst_src
from pipeline.tracking.iou_tracking import iou_tracking
from pipeline.tracking.gnn_tracking import gnn_tracking
from pipeline.tracking.man_tracking import man_tracking
from pipeline.settings.Setting_Classes import Settings

@dataclass
class TrackingModule(BaseModule):
    ## Attributes from the BaseModule class:
        # input_folder: PathLike | list[PathLike]
        # exp_obj_lst: list[Experiment] = field(init=False)
        # optimization: bool = False
    
    def track_from_settings(self, settings: dict)-> list[Experiment]:
        # If optimization is set, then process only the first experiment
        self.optimization = settings['optimization']
        
        # Track the cells based on the settings
        sets = Settings(settings)
        if not hasattr(sets,'tracking'):
            print("No tracking settings found")
            self.save_as_json()
            return self.exp_obj_lst
        sets = sets.tracking

        if hasattr(sets,'iou_track'):
            self.iou_tracking(**sets.iou_track)
            return self.exp_obj_lst
    
        if hasattr(sets,'gnn_track'):
            self.exp_obj_lst = self.gnn_tracking(**sets.gnn_track)
            self.save_as_json()
            return self.exp_obj_lst
        
        if hasattr(sets,'man_track'):
            self.exp_obj_lst = self.man_tracking(**sets.man_track)
            self.save_as_json()
            return self.exp_obj_lst
    
    @staticmethod
    def _iou_tracking(exp_obj: Experiment, channel_to_track: str, mask_fold_src: PathLike, stitch_thres_percent: float, shape_thres_percent: float, overwrite: bool, mask_appear: int, copy_first_to_start: bool, copy_last_to_end: bool)-> None:
        """Function to track the segemented cells using IOU method. This function mainly apply to cells that are not moving much between frames. If both copy_first_to_start (copy the first mask appearance to the start of the sequence) and copy_last_to_end (copy the last mask appearance to the end of the sequence) are True, then the incomplete tracks will be removed, else all tracks will be returned.
        
        Args:
            exp_obj (Experiment): Experiment object.
            channel_to_track (str): Channel to track.
            mask_fold_src (PathLike): Folder containing the segemented masks.
            stitch_thres_percent (float): Percentage of the mask overlap to consider as the same cell.
            shape_thres_percent (float): Percentage of the mask shape similarity to consider as the same cell.
            overwrite (bool): Overwrite the previous tracking.
            mask_appear (int): Number of appearance of the mask to consider a track as valid.
            copy_first_to_start (bool): Copy the first mask appearance to the start of the sequence
            copy_last_to_end (bool): Copy the last mask appearance to the end of the sequence"""
        
        # Activate branch
        exp_obj.tracking.is_iou_tracking = True
        
        # Get the mask paths and metadata
        mask_fold_src,mask_paths = seg_mask_lst_src(exp_obj,mask_fold_src)
        um_per_pixel = exp_obj.analysis.um_per_pixel
        finterval = exp_obj.analysis.interval_sec
            
        # Run IOU tracking
        iou_tracking(mask_paths,channel_to_track,stitch_thres_percent,shape_thres_percent,overwrite,mask_appear,copy_first_to_start,copy_last_to_end,um_per_pixel=um_per_pixel,finterval=finterval)
        
        # Save settings
        exp_obj.tracking.iou_tracking[channel_to_track] = {'fold_src':mask_fold_src,
                                                        'stitch_thres_percent':stitch_thres_percent,
                                                        'shape_thres_percent':shape_thres_percent,
                                                        'mask_appear':mask_appear}
        exp_obj.save_as_json()
    
    def iou_tracking(self, channel_to_track: str | list[str], mask_fold_src: PathLike = "", stitch_thres_percent: float=0.75, shape_thres_percent: float=0.2, overwrite: bool=False, mask_appear: int=5, copy_first_to_start: bool=True, copy_last_to_end: bool=True)-> list[Experiment]:
        if isinstance(channel_to_track, str):
            print(f"\n-> Tracking images with IOU")
            
            self._loop_over_exp(self._iou_tracking,channel_to_track=channel_to_track,mask_fold_src=mask_fold_src,stitch_thres_percent=stitch_thres_percent,shape_thres_percent=shape_thres_percent,overwrite=overwrite,mask_appear=mask_appear,copy_first_to_start=copy_first_to_start,copy_last_to_end=copy_last_to_end)
        
        if isinstance(channel_to_track,list):
            for channel in channel_to_track:
                self.exp_obj_lst = self.iou_tracking(channel,mask_fold_src,stitch_thres_percent,shape_thres_percent,overwrite,mask_appear,copy_first_to_start,copy_last_to_end)
            return self.exp_obj_lst
        
    def gnn_tracking(self, channel_to_track: str | list[str], max_travel_dist: int,img_fold_src: PathLike ="", model: str='neutrophil', mask_fold_src: PathLike ="", morph: bool=False, decision_threshold: float=0.5, mask_appear: int=2, manual_correct: bool=False, trim_incomplete_tracks: bool= False, overwrite: bool=False) -> list[Experiment]:
        # If optimization is set, then process only the first experiment
        exp_obj_lst = self.exp_obj_lst.copy()[:1] if self.optimization else self.exp_obj_lst
        
        if isinstance(channel_to_track, str):
            return gnn_tracking(exp_obj_lst,channel_to_track,model,max_travel_dist,overwrite,img_fold_src,mask_fold_src,morph,mask_appear,decision_threshold,manual_correct,trim_incomplete_tracks)
        
        if isinstance(channel_to_track,list):
            for channel in channel_to_track:
                self.exp_obj_lst = self.gnn_tracking(channel,model,max_travel_dist,overwrite,img_fold_src,mask_fold_src,morph,mask_appear,decision_threshold,manual_correct,trim_incomplete_tracks)
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