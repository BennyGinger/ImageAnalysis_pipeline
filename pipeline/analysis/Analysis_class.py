from __future__ import annotations
from dataclasses import dataclass
from os import PathLike
from image_handeling.Base_Module_Class import BaseModule
from image_handeling.Experiment_Classes import Experiment, init_from_json
from .channel_data import extract_channel_data
from settings.Setting_Classes import Settings

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
    
    def extract_channel_data(self, channel_name: str | list[str], img_fold_src: PathLike = "", overwrite: bool=False)-> list[Experiment]:
        pass