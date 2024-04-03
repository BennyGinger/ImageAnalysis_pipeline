from __future__ import annotations
from dataclasses import dataclass, field
from os import PathLike
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
    
    #TODO: add methods to change channel names