from __future__ import annotations
from dataclasses import dataclass, field
from os import PathLike, walk
from os.path import join
from typing import Callable
from pipeline.utilities.pipeline_utility import progress_bar, pbar_desc
from .Experiment_Classes import Experiment, init_from_json


@dataclass
class BaseModule:
    input_folder: PathLike | list[PathLike]
    exp_obj_lst: list[Experiment] = field(default_factory=list)
    optimization: bool = False
    
    def __post_init__(self)-> None:
        if self.exp_obj_lst:
            print(f"\n\033[92m===== Loading the {self.__class__.__name__} Module =====\033[0m")
            return
        
        # Initialize the experiment list
        print(f"\n\033[92m===== Initializing the {self.__class__.__name__} Module =====\033[0m")
        self.init_exp_obj()
    
    def init_exp_obj(self)-> None:
        # Initialize the experiment list
        jsons_path = self.gather_all_json_path()
        self.exp_obj_lst = [init_from_json(json_path) for json_path in jsons_path]
    
    def change_attribute(self, attribute: str, value: any)-> list[Experiment]:
        for exp_obj in self.exp_obj_lst:
            exp_obj.set_attribute(attribute,value)
            exp_obj.save_as_json()
        return self.exp_set_list
    
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
    
    def save_as_json(self)-> None:
        for exp_obj in self.exp_obj_lst:
            exp_obj.save_as_json()
    
    def _loop_over_exp(self, func: Callable, **kwargs)-> None:
        # If optimization is set, then process only the first experiment
        exp_obj_lst = self.exp_obj_lst.copy()[:1] if self.optimization else self.exp_obj_lst
        
        # Loop over all the experiments and apply the function
        for exp_obj in progress_bar(exp_obj_lst,
                            desc=pbar_desc("Experiments"),
                            colour='blue'):
            func(exp_obj,**kwargs)
       
    @staticmethod
    def get_json_path(folder: str)-> list[str]:
        # Get the path of all the nd2 files in all subsequent folders/subfolders and exp_dict if available
        jsons_path = []
        for root , _, files in walk(folder):
            for f in files:
                # Look for all files with selected extension and that are not already processed 
                if f == 'exp_settings.json':
                    jsons_path.append(join(root,f))
        return sorted(jsons_path)




    #TODO: add methods to change channel names