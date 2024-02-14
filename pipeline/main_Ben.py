from __future__ import annotations
from time import time
# TODO: fix module path 
from Experiment_Classes import Experiment
from pre_process.PreProcess_Class import PreProcess
from settings.pre_process import settings

INPUT_FOLDER = settings['input_folder']

def change_attribute(exp_set_list: list[Experiment], attribute: str, value: any)-> list[Experiment]:
    for exp_set in exp_set_list:
        exp_set.set_attribute(attribute,value)
        exp_set.save_as_json()
    return exp_set_list

if __name__ == "__main__":

    t1 = time()
    
    exp_list = PreProcess(INPUT_FOLDER,**settings['init']).process_from_settings(settings)
    
    t2 = time()
    if t2-t1<60: print(f"Time to process: {round(t2-t1,ndigits=3)} sec\n")
    else: print(f"Time to process: {round((t2-t1)/60,ndigits=1)} min\n")