settings = {
    "input_folder": '/home/Fabian/ImageData/CalciumImmunCell/Macrophage/Amputation',
    "init":{"active_channel_list": ['RFP','GFP'],
            'full_channel_list':['RFP','GFP'],
            "overwrite": False},
    
    
    "bg_sub": (False,
                {"sigma": 0.0,
                "size": 7,
                "overwrite": False}),
    
    "chan_shift": (False,
                    {"reg_channel": "RFP",
                    "reg_mtd": "translation",
                    "overwrite": False}),
    
    "frame_shift": (False,
                {"reg_channel": "RFP",
                "reg_mtd": "rigid_body",
                "img_ref": "mean",
                "overwrite": False}),
    
    "blur": (False,
            {"kernel": (15,15),
            "sigma": 5,
            "img_fold_src": "",
            "overwrite": False}),

    "cellpose": (True,
                {"channel_to_seg":"RFP", #BUG channel should be controlled if it exists in the channel list
                "model_type": "/home/Fabian/Models/Cellpose/twoFishMacrophage", #cyto2_cp3, cyto3, /home/Fabian/Models/Cellpose/twoFishMacrophage
                "diameter": 15.0,
                "flow_threshold": 0.0,
                "cellprob_threshold":-3,
                "overwrite": False,
                "img_fold_src": "",
                "process_as_2D": True,
                "save_as_npy": False,
                "nuclear_marker": "",
                "normalize":{"percentile":[20,99]}}),
    
    "threshold": (False,
                {"channel_to_seg":"RFP",
                "overwrite": False,
                "manual_threshold": None,
                "img_fold_src": "",}),
    
    "iou_track": (False,
                  {"channel_to_track":"YFP", #BUG channel should be controlled if it exists in the channel list
                   "img_fold_src": "",
                   "stitch_thres_percent": 0.1,
                   "shape_thres_percent": 0.8,
                   "mask_appear":90,
                   "copy_first_to_start": True, 
                   "copy_last_to_end": True,
                   "overwrite":True}),
    
    "gnn_track": (True,
                  {"channel_to_track":"RFP",
                   "img_fold_src": "",
                   "model":"neutrophil", #neutrophil, neutrophil_old, Fluo-C2DL-Huh7, Fluo-N2DH-SIM+, Fluo-N2DL-HeLa, Fluo-N3DH-SIM+ (implement from server first!), PhC-C2DH-U373
                   "mask_fold_src": "",
                   "morph":False, # not implemented yet
                   'min_cell_size': 15,
                   'decision_threshold': 0.5, #between 0-1, 1=more interrupted tracks, 0= more tracks gets connected, checks for the confidence of the model for the connection of two cells
                   "mask_appear":5, # not implemented yet
                   "manual_correct":True,
                   "overwrite":True}),
    
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
    
    "extract_data": (False,
                        {"img_fold_src": "",
                         "drawing_label":"wound",
                         "channel_show":"GFP",
                        "overwrite": True}),
}