from __future__ import annotations
from dataclasses import dataclass, field
from os import PathLike, walk, sep
from os.path import join
from typing import Any
from Experiment_Classes import Experiment, init_from_json
from .cp_segmentation import cellpose_segmentation

@dataclass
class Segmentation:
    input_folder: str | list[str]
    experiment_list: list[Experiment] = field(default_factory=list)
    
    def __post_init__(self)-> None:
        if self.experiment_list:
            print("Loading the Segmentation Module")
            return
        
        # Initialize the experiment list
        print("Initializing the Segmentation Module")
        jsons_path = gather_all_json_path(self.input_folder)
        
        self.experiment_list = [init_from_json(json_path) for json_path in jsons_path]
            
    def segment_from_settings(self, settings: dict)-> list[Experiment]:
        
        return self.experiment_list
    
    def cellpose(self, channel_to_seg: str | list[str], model_type: str | PathLike = 'cyto3', diameter: float = 60, flow_threshold: float = 0.4, 
                 cellprob_threshold: float = 0, cellpose_overwrite: bool = False, img_fold_src: str = "", as_2D: bool = False, as_npy: bool = False,
                 nuclear_marker: str = "",**kwargs: Any)-> list[Experiment]:
        
        if isinstance(channel_to_seg,str):
            return cellpose_segmentation(self.experiment_list,channel_to_seg,model_type,diameter,flow_threshold,
                                         cellprob_threshold,cellpose_overwrite,img_fold_src,as_2D,as_npy,nuclear_marker,**kwargs)
        if isinstance(channel_to_seg,list):
            for channel in channel_to_seg:
                self.cellpose(channel,model_type,diameter,flow_threshold,cellprob_threshold,cellpose_overwrite,img_fold_src,as_2D,as_npy,nuclear_marker,**kwargs)
                
                
                
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
    
    