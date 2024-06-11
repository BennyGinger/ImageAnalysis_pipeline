from __future__ import annotations
from dataclasses import dataclass
from os import PathLike
from typing import Any
from pipeline.utilities.Base_Module_Class import BaseModule
from pipeline.utilities.Experiment_Classes import Experiment
from pipeline.segmentation.cp_segmentation import cellpose_segmentation
from pipeline.segmentation.segmentation import threshold
from pipeline.settings.Setting_Classes import Settings
from pipeline.utilities.data_utility import img_list_src

@dataclass
class SegmentationModule(BaseModule):
    ## Attributes from the BaseModule class:
        # input_folder: PathLike | list[PathLike]
        # exp_obj_lst: list[Experiment] = field(init=False)
        # optimization: bool = False
            
    def segment_from_settings(self, settings: dict)-> list[Experiment]:
        # If optimization is set, then process only the first experiment
        self.optimization = settings['optimization']

        # Segment the cells based on the settings
        sets = Settings(settings)
        if not hasattr(sets,'segmentation'):
            print("\n\033[93mNo segmentation settings found =====\033[0m")
            self.save_as_json()
            return self.exp_obj_lst
        sets = sets.segmentation
        # Run the segmentation processes
        print("\n\033[93mSegmentation process started =====\033[0m")
        if hasattr(sets,'cellpose'):
            self.cellpose(**sets.cellpose)
        if hasattr(sets,'threshold'):
            self.thresholding(**sets.threshold)
        print("\n\033[93mSegmentation done =====\033[0m")
        self.save_as_json()
        return self.exp_obj_lst
    
    @staticmethod
    def _cellpose(exp_obj: Experiment, channel_to_seg: str, model_type: str, diameter: float, flow_threshold: float, cellprob_threshold: float, overwrite: bool, img_fold_src: str, process_as_2D: bool, save_as_npy: bool, **kwargs)-> None:
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
    
    def cellpose(self, channel_to_seg: str | list[str], model_type: str | PathLike = 'cyto2', diameter: float = 60, flow_threshold: float = 0.4, cellprob_threshold: float = 0, overwrite: bool = False, img_fold_src: str = "", process_as_2D: bool = False, save_as_npy: bool = False,**kwargs)-> None:
        if isinstance(channel_to_seg,str):
            print(f"\n-> Segmenting images with cellpose")
            
            self._loop_over_exp(self._cellpose,channel_to_seg=channel_to_seg,model_type=model_type,diameter=diameter,flow_threshold=flow_threshold,cellprob_threshold=cellprob_threshold,overwrite=overwrite,img_fold_src=img_fold_src,process_as_2D=process_as_2D,save_as_npy=save_as_npy,**kwargs)
        
        if isinstance(channel_to_seg,list):
            for channel in channel_to_seg:
                self.cellpose(channel,model_type,diameter,flow_threshold,cellprob_threshold,
                              overwrite,img_fold_src,process_as_2D,save_as_npy,**kwargs)
       
    @staticmethod
    def _thresholding(exp_obj: Experiment, channel_to_seg: str, overwrite: bool, manual_threshold: int, img_fold_src: str)-> None:
        # Activate branch
        exp_obj.segmentation.is_threshold = True
        # Get the image paths and metadata
        img_fold_src,img_paths = img_list_src(exp_obj,img_fold_src)
        um_per_pixel = exp_obj.analysis.um_per_pixel
        finterval = exp_obj.analysis.interval_sec
        
        # Run thresholding
        exp_obj.segmentation.threshold_seg[channel_to_seg] = threshold(img_paths,channel_to_seg,overwrite,manual_threshold,um_per_pixel,finterval)
        exp_obj.save_as_json()
        return
    
    def thresholding(self, channel_to_seg: str | list[str], overwrite: bool=False, manual_threshold: int=None, img_fold_src: str="")-> None:
        
        if isinstance(channel_to_seg,str):
            print(f"\n-> Thresholding images")
            
            self._loop_over_exp(self._thresholding,channel_to_seg=channel_to_seg,overwrite=overwrite,manual_threshold=manual_threshold,img_fold_src=img_fold_src)
        
        if isinstance(channel_to_seg,list):
            for channel in channel_to_seg:
                self.exp_obj_lst = self.thresholding(channel,overwrite,manual_threshold,img_fold_src)
            return
        
                
                
