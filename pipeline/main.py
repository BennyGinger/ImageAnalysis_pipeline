from __future__ import annotations
# Force the multiprocessing to to start with new interpreter, as pytorch by default will pre-configure the interpreter, which will 'hog' CPU usage
from multiprocessing import set_start_method
set_start_method("spawn",force=True)
from time import time
from pipeline.pre_process.PreProcess_Class import PreProcess
from pipeline.settings.settings_dict import settings
from pipeline.segmentation.Segmentation_Class import Segmentation
from pipeline.tracking.Tracking_Class import Tracking
from pipeline.analysis.Analysis_class import Analysis
INPUT_FOLDER = settings['input_folder']


if __name__ == "__main__":

    t1 = time()
    exp_list = PreProcess(INPUT_FOLDER,**settings['init']).process_from_settings(settings)
    exp_list = Segmentation(INPUT_FOLDER,exp_list).segment_from_settings(settings)
    exp_list = Tracking(INPUT_FOLDER,exp_list).track_from_settings(settings)
    exp_list = Analysis(INPUT_FOLDER,exp_list).extract_data()
    
    t2 = time()
    if t2-t1<60: print(f"Time to process: {round(t2-t1,ndigits=3)} sec\n")
    else: print(f"Time to process: {round((t2-t1)/60,ndigits=1)} min\n")