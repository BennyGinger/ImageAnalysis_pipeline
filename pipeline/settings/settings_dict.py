settings = {
    "input_folder": '/home/Test_images/nd2/Run2',
    "init":{"active_channel_list": ['GFP','RFP'],
            'full_channel_list':[],
            "overwrite": True},
    
    
    "bg_sub": (True,
                {"sigma": 0.0,
                "size": 7,
                "overwrite": False}),
    
    "chan_shift": (False,
                    {"reg_channel": "RFP",
                    "reg_mtd": "rigid_body",
                    "overwrite": False}),
    
    "frame_shift": (False,
                {"reg_channel": "RFP",
                "reg_mtd": "rigid_body",
                "img_ref": "previous",
                "overwrite": True}),
    
    "blur": (False,
            {"kernel": (15,15),
            "sigma": 5,
            "img_fold_src": "",
            "overwrite": False}),

    "cellpose": (False,
                {"channel_to_seg":"RFP",
                "model_type": "cyto3",
                "diameter": 60.0,
                "flow_threshold": 0.4,
                "cellprob_threshold": 0.0,
                "overwrite": False,
                "img_fold_src": "",
                "process_as_2D": True,
                "save_as_npy": False,
                "nuclear_marker": "",}),
    
    "threshold": (False,
                {"channel_to_seg":"RFP",
                "overwrite": True,
                "manual_threshold": None,
                "img_fold_src": "",}),
    
    "iou_track": (True,
                  {"channel_to_track":"RFP",
                   "img_fold_src": "",
                   "stitch_thres_percent": 0.75,
                   "shape_thres_percent": 0.2,
                   "n_mask":5,
                   "overwrite":False}),
}