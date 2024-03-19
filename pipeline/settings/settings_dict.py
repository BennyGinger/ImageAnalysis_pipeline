settings = {
    "input_folder": '/home/Test_images/tiff/Run2',
    "init":{"active_channel_list": ['GFP','RFP'],
            'full_channel_list':[],
            "overwrite": True},
    
    
    "run_bg_sub": True,
    "bg_sub": {"sigma": 0.0,
               "size": 7,
               "overwrite": False},
    
    "run_channel_reg": False,
    "chan_shift": {"reg_channel": "RFP",
                   "reg_mtd": "rigid_body",
                   "overwrite": False},
    
    "run_frame_reg": False,
    "frame_shift": {"reg_channel": "RFP",
                 "reg_mtd": "rigid_body",
                 "img_ref": "previous",
                 "overwrite": True},
    
    "run_blur": False,
    "blur": {"kernel": (15,15),
             "sigma": 5,
             "img_fold_src": "",
             "overwrite": False},

    "run_cellpose": False,
    "cellpose": {"channel_to_seg":"RFP",
                 "model_type": "cyto3",
                 "diameter": 60.0,
                 "flow_threshold": 0.4,
                 "cellprob_threshold": 0.0,
                 "overwrite": True,
                 "img_fold_src": "",
                 "process_as_2D": True,
                 "save_as_npy": False,
                 "nuclear_marker": "",},
    
    "run_threshold": False,
    "threshold": {"channel_to_seg":"RFP",
                  "overwrite": True,
                  "manual_threshold": None,
                  "img_fold_src": "",},
}