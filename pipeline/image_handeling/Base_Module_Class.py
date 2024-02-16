from __future__ import annotations
from dataclasses import dataclass, field
from os import PathLike
from .Experiment_Classes import Experiment


@dataclass
class BaseModule:
    input_folder: PathLike | list[PathLike]
    experiment_list: list[Experiment] = field(init=False)
    
    def __post_init__(self)-> None:
        if self.experiment_list:
            print(f"Loading the {self.__class__.__name__} Module")
            return
        
        # Initialize the experiment list
        print(f"Initializing the {self.__class__.__name__} Module")
    
    def change_attribute(self, attribute: str, value: any)-> list[Experiment]:
        for exp_set in self.exp_set_list:
            exp_set.set_attribute(attribute,value)
            exp_set.save_as_json()
        return self.exp_set_list