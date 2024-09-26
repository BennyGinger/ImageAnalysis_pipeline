from __future__ import annotations
# Force the multiprocessing to to start with new interpreter, as pytorch by default will pre-configure the interpreter, which will 'hog' CPU usage
from multiprocessing import set_start_method
set_start_method("spawn",force=True)

import pandas as pd
from pipeline.image_extraction.ImageExtract_Module import ImageExtractionModule
from pipeline.pre_process.PreProcess_Module import PreProcessModule
from pipeline.segmentation.Segmentation_Module import SegmentationModule
from pipeline.tracking.Tracking_Class import TrackingModule
from pipeline.analysis.Analysis_class import AnalysisModule

def run_pipeline(settings: dict)-> pd.DataFrame:
    input_folder = settings['input_folder']
    exp_list = ImageExtractionModule(input_folder,**settings['init']).extract_img_seq()
    
    exp_list = PreProcessModule(input_folder,exp_list).process_from_settings(settings)
    
    # For batch, draw the mask asap after registration, so no need to wait the all process
    if 'draw_mask' in settings and settings["draw_mask"][0]:
        AnalysisModule(input_folder,exp_list).draw_wound_mask(**settings["draw_mask"][1])
    
    exp_list = SegmentationModule(input_folder).segment_from_settings(settings)
    exp_list = TrackingModule(input_folder,exp_list).track_from_settings(settings)
    
    if settings["extract_data"][0]:
        master_df = AnalysisModule(input_folder,exp_list).create_master_df(**settings["extract_data"][1])
    else:
        master_df = pd.DataFrame()
    return master_df

def run_preprocess(settings: dict)-> None:
    input_folder = settings['input_folder']
    exp_list = ImageExtractionModule(input_folder,**settings['init']).extract_img_seq()
    PreProcessModule(input_folder,exp_list).process_from_settings(settings)

def run_segmentation(settings: dict)-> None:
    input_folder = settings['input_folder']
    SegmentationModule(input_folder).segment_from_settings(settings)

def run_tracking(settings: dict)-> None: 
    input_folder = settings['input_folder']
    TrackingModule(input_folder).track_from_settings(settings)

def run_analysis(settings: dict)-> pd.DataFrame:
    input_folder = settings['input_folder']
    return AnalysisModule(input_folder).analyze_from_settings(settings)

def reset_overwrite(settings: dict)-> None:
    for k,v in settings.items():
        if k == "optimization":
            settings[k] = False
        
        if k == "overwrite":
            settings[k] = False
        
        if isinstance(v, dict):
            reset_overwrite(v)
        elif isinstance(v, tuple):
            reset_overwrite(v[1])


if __name__ == "__main__":
    from time import time

    settings = {
    "input_folder": '/home/Test_images/nd2/Run4',
    
    "optimization": False,
    
    "init":{"active_channel_list": ['GFP','RFP'],
            'full_channel_list': ["DAPI","GFP","RFP","iRed"],
            "overwrite": False},
    
    "bg_sub": (True,
                {"overwrite": False}),
    
    "chan_shift": (True,
                    {"reg_channel": "RFP",
                    "reg_mtd": "rigid_body",
                    "overwrite": False}),
    
    "frame_shift": (True,
                {"reg_channel": "RFP",
                "reg_mtd": "rigid_body",
                "img_ref": "first",
                "overwrite": False}),
    
    "blur": (True,
            {"sigma": 2,
            "overwrite": True}),

    "cellpose": (True,
                {"channel_to_seg":["RFP","GFP"], 
                "model_type": "cyto2", #cyto2_cp3, cyto3, /home/Fabian/Models/Cellpose/twoFishMacrophage
                "diameter": 65,
                "flow_threshold": 0.6,
                "cellprob_threshold":0,
                "process_as_2D": True,
                "overwrite": False,}),
    
    "threshold": (False,
                {"channel_to_seg":"RFP",
                "manual_threshold": None,
                "img_fold_src": "",
                "overwrite": False,}),
    
    "iou_track": (True,
                  {"channel_to_track":["RFP","GFP"], 
                   "stitch_thres_percent": 0.5,
                   "shape_thres_percent": 0.95,
                   "mask_appear":5,
                   "copy_first_to_start": True, 
                   "copy_last_to_end": True,
                   "overwrite":False}),
    
    "gnn_track": (False,                         #not working: Fluo-C2DL-Huh7
                  {"channel_to_track": "RFP",
                   "model": "PhC-C2DH-U373", #neutrophil, Fluo-N2DH-SIM+, Fluo-N2DL-HeLa, Fluo-N3DH-SIM+ (implement from server first!), PhC-C2DH-U373
                   'decision_threshold': 0.4, #between 0-1, 1=more interrupted tracks, 0= more tracks gets connected, checks for the confidence of the model for the connection of two cells
                   'max_travel_dist': 10,
                   "manual_correct": True,
                   "trim_incomplete_tracks": True,
                   "overwrite": False}),
    
    "man_track": (False,
                  {"channel_to_track":"BF",
                   "track_seg_mask": False,
                   "mask_fold_src": "",
                   "csv_name": "",
                   "radius": 5,
                   'copy_first_to_start': True,
                   'copy_last_to_end': True,
                   'mask_appear': 2,
                   "dilate_value":20,
                   "process_as_2D":True,
                   "overwrite":True}),
    
    "draw_mask": (False,
                  {"mask_label": "laser", # str or list[str]
                   "channel_show": "RFP",
                   "overwrite": False}),
    
    "extract_data": (True,
                  {"num_chunks": 3,
                   "do_diff": True,
                   "overwrite": True}),
    }
    
    t1 = time()
    input_folder = settings['input_folder']
    master_df = run_pipeline(settings)
    
    t2 = time()
    if t2-t1<60: print(f"Time to process: {round(t2-t1,ndigits=3)} sec\n")
    else: print(f"Time to process: {round((t2-t1)/60,ndigits=1)} min\n")
    