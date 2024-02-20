from __future__ import annotations
from time import time
from pre_process.PreProcess_Class import PreProcess
from segmentation.Segmentation_Class import Segmentation
from settings.settings_dict import settings

INPUT_FOLDER = settings['input_folder']



if __name__ == "__main__":

    t1 = time()
    
    exp_list = PreProcess(INPUT_FOLDER,**settings['init']).process_from_settings(settings)
    # exp_list = Segmentation(INPUT_FOLDER,exp_list).segment_from_settings(settings)
    
    t2 = time()
    if t2-t1<60: print(f"Time to process: {round(t2-t1,ndigits=3)} sec\n")
    else: print(f"Time to process: {round((t2-t1)/60,ndigits=1)} min\n")