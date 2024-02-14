from __future__ import annotations
from dataclasses import dataclass, field
from os import walk, sep
from os.path import join
from Experiment_Classes import Experiment, init_from_json

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
        jsons_path = self.gather_all_json_path()
        
        self.experiment_list = [init_from_json(json_path) for json_path in jsons_path]
    
    def gather_all_json_path(self)-> list[str]:
        print(f"\nSearching for 'exp_settings.json' files in {self.input_folder}")
        # Get the path of all the json files in all subsequent folders/subfolders
        if isinstance(self.input_folder,str):
            return self.get_json_path(self.input_folder)
        
        if isinstance(self.input_folder,list):
            jsons_path = []
            for folder in self.input_folder:
                jsons_path.extend(self.get_json_path(folder))
            return jsons_path
            
    @staticmethod    
    def get_json_path(folder: str)-> list[str]:
        # Get the path of all the nd2 files in all subsequent folders/subfolders and exp_dict if available
        jsons_path = []
        for root , _, files in walk(folder):
            for f in files:
                # Look for all files with selected extension and that are not already processed 
                if f == 'exp_settings.json':
                    jsons_path.append(join(sep,root+sep,f))
        return sorted(jsons_path)
    
    def segment_from_settings(self, settings: dict)-> list[Experiment]:
        
        return self.experiment_list