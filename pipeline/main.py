from __future__ import annotations
# Force the multiprocessing to to start with new interpreter, as pytorch by default will pre-configure the interpreter, which will 'hog' CPU usage
from multiprocessing import set_start_method
set_start_method("spawn",force=True)
from time import time
import pandas as pd
from pipeline.pre_process.PreProcess_Class import PreProcessModule
from pipeline.settings.settings_dict import settings
from pipeline.segmentation.Segmentation_Class import SegmentationModule
from pipeline.tracking.Tracking_Class import TrackingModule
from pipeline.analysis.Analysis_class import AnalysisModule

def run_pipeline(settings: dict)-> pd.DataFrame:
    
    input_folder = settings['input_folder']
    exp_list = PreProcessModule(input_folder,
                                **settings['init']).process_from_settings(settings)
    exp_list = SegmentationModule(input_folder,exp_list).segment_from_settings(settings)
    exp_list = TrackingModule(input_folder,exp_list).track_from_settings(settings)
    master_df = AnalysisModule(input_folder,exp_list).analyze_from_settings(settings)
    return master_df

if __name__ == "__main__":

    
    t1 = time()
    # exp_list = PreProcess(INPUT_FOLDER,**settings['init']).process_from_settings(settings)
    # exp_list = Segmentation(INPUT_FOLDER,exp_list).segment_from_settings(settings)
    # exp_list = Tracking(INPUT_FOLDER,exp_list).track_from_settings(settings)
    # master_df = Analysis(INPUT_FOLDER,exp_list).analyze_from_settings(settings)
    master_df = run_pipeline(settings)
    
    t2 = time()
    if t2-t1<60: print(f"Time to process: {round(t2-t1,ndigits=3)} sec\n")
    else: print(f"Time to process: {round((t2-t1)/60,ndigits=1)} min\n")