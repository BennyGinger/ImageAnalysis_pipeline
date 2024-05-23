settings = {
    "input_folder": '/home/Test_images/nd2/Run2_test',
    "init":{"active_channel_list": ['GFP','RFP'],
            'full_channel_list':['GFP','RFP'],
            "overwrite": True},
    
    "bg_sub": (True,
                {"sigma": 0.0,
                "size": 7,
                "overwrite": False}),
    
    "chan_shift": (True,
                    {"reg_channel": "RFP",
                    "reg_mtd": "rigid_body",
                    "overwrite": False}),
    
    "frame_shift": (False,
                {"reg_channel": "BF",
                "reg_mtd": "rigid_body",
                "img_ref": "previous",
                "overwrite": True}),
    
    "blur": (False,
            {"kernel": (15,15),
            "sigma": 5,
            "overwrite": False}),

    "cellpose": (False,
                {"channel_to_seg":"RFP", #BUG channel should be controlled if it exists in the channel list
                "model_type": "cyto2", #cyto2_cp3, cyto3, /home/Fabian/Models/Cellpose/twoFishMacrophage
                "diameter": 60.0,
                "flow_threshold": 0.5,
                "cellprob_threshold":0,
                "process_as_2D": True,
                "save_as_npy": False,
                "overwrite": True,}),
    
    "threshold": (False,
                {"channel_to_seg":"RFP",
                "manual_threshold": None,
                "img_fold_src": "",
                "overwrite": False,}),
    
    "iou_track": (False,
                  {"channel_to_track":"RFP", #BUG channel should be controlled if it exists in the channel list
                   "img_fold_src": "",
                   "stitch_thres_percent": 0.1,
                   "shape_thres_percent": 0.8,
                   "mask_appear":5,
                   "copy_first_to_start": True, 
                   "copy_last_to_end": True,
                   "overwrite":True}),
    
    "gnn_track": (False,                         #not working: Fluo-C2DL-Huh7
                  {"channel_to_track":"RFP",
                   "img_fold_src": "",
                   "model":"PhC-C2DH-U373", #neutrophil, Fluo-N2DH-SIM+, Fluo-N2DL-HeLa, Fluo-N3DH-SIM+ (implement from server first!), PhC-C2DH-U373
                   "mask_fold_src": "",
                   "morph":False, # not implemented yet
                   'decision_threshold': 0.4, #between 0-1, 1=more interrupted tracks, 0= more tracks gets connected, checks for the confidence of the model for the connection of two cells
                   'max_travel_dist':50,
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
    
    "draw_mask": (False,
                  {"mask_label": "wound", # str or list[str]
                   "channel_show": "RFP",
                   "overwrite": True}),
    
    "extract_data": (False,
                  {"img_fold_src": "",
                   "overwrite": True}),
}