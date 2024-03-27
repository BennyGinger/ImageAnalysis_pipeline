from __future__ import annotations
from dataclasses import dataclass, field
from os import PathLike, walk
from os.path import join
from .Experiment_Classes import Experiment


@dataclass
class BaseModule:
    input_folder: PathLike | list[PathLike]
    exp_obj_lst: list[Experiment] = field(default_factory=list)
    
    def __post_init__(self)-> None:
        if self.exp_obj_lst:
            print(f"\nLoading the {self.__class__.__name__} Module")
            return
        
        # Initialize the experiment list
        print(f"\nInitializing the {self.__class__.__name__} Module")
    
    def change_attribute(self, attribute: str, value: any)-> list[Experiment]:
        for exp_obj in self.exp_obj_lst:
            exp_obj.set_attribute(attribute,value)
            exp_obj.save_as_json()
        return self.exp_set_list
    
    def gather_all_json_path(self)-> list[str]:
        print(f"\nSearching for 'exp_settings.json' files in {self.input_folder}")
        # Get the path of all the json files in all subsequent folders/subfolders
        if isinstance(self.input_folder,str):
            return get_json_path(self.input_folder)
        
        if isinstance(self.input_folder,list):
            jsons_path = []
            for folder in self.input_folder:
                jsons_path.extend(get_json_path(folder))
            return jsons_path
    
    def save_as_json(self)-> None:
        for exp_obj in self.exp_obj_lst:
            exp_obj.save_as_json()
       
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