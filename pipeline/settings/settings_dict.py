settings = {
    "input_folder": '/home/Fabian/ImageData/TrackingTestFiles/PipelineTest',
    "init":{"active_channel_list": ['RFP','GFP'],
            'full_channel_list':[],
            "overwrite": False},
    
    
    "bg_sub": (True,
                {"sigma": 0.0,
                "size": 7,
                "overwrite": False}),
    
    "chan_shift": (True,
                    {"reg_channel": "RFP",
                    "reg_mtd": "translation",
                    "overwrite": False}),
    
    "frame_shift": (True,
                {"reg_channel": "RFP",
                "reg_mtd": "rigid_body",
                "img_ref": "previous",
                "overwrite": False}),
    
    "blur": (False,
            {"kernel": (15,15),
            "sigma": 5,
            "img_fold_src": "",
            "overwrite": False}),

    "cellpose": (True,
                {"channel_to_seg":"RFP",
                "model_type": "cyto3",
                "diameter": 15.0,
                "flow_threshold": 0.5,
                "cellprob_threshold": 0.0,
                "overwrite": False,
                "img_fold_src": "",
                "process_as_2D": True,
                "save_as_npy": False,
                "nuclear_marker": "",}),
    
    "threshold": (False,
                {"channel_to_seg":"RFP",
                "overwrite": False,
                "manual_threshold": None,
                "img_fold_src": "",}),
    
    "iou_track": (False,
                  {"channel_to_track":"RFP",
                   "img_fold_src": "",
                   "stitch_thres_percent": 0.1,
                   "shape_thres_percent": 0.8,
                   "mask_appear":5,
                   "copy_first_to_start": True, 
                   "copy_last_to_end": True,
                   "overwrite":True}),
    
    "gnn_track": (True,
                  {"channel_to_track":"RFP",
                   "img_fold_src": "",
                   "model":"Fluo-N2DL-HeLa", #neutrophil, neutrophil_old, Fluo-C2DL-Huh7, Fluo-N2DH-SIM+, Fluo-N2DL-HeLa, Fluo-N3DH-SIM+, PhC-C2DH-U373
                   "mask_fold_src": "",
                   "morph":False, # not implemented yet
                   'min_cell_size': 15,
                   'decision_threshold': 0.5,
                   "mask_appear":5, # not implemented yet
                   "overwrite":True}),
    
    "man_track": (False,
                  {"channel_to_track":"RFP",
                   "track_seg_mask": False,
                   "mask_fold_src": "",
                   "csv_name": "",
                   "radius": 5,
                   'morph': True,
                   'mask_appear': 2,
                   "dilate_value":20,
                   "overwrite":True}),
}