settings = {
    "input_folder": '/home/Test_images/nd2/Run2',
    "init":{"active_channel_list": ['GFP','RFP'],
            'full_channel_list':[],
            "overwrite": True},
    
    
    "run_bg_sub": True,
    "bg_sub": {"sigma": 0.0,
               "size": 7,
               "overwrite": False},
    
    "run_chan_shift": True,
    "chan_shift": {"reg_channel": "RFP",
                   "reg_mtd": "rigid_body",
                   "overwrite": False,
                   "first_img_only": True},
    
    "run_register": True,
    "register": {"reg_channel": "RFP",
                 "reg_mtd": "rigid_body",
                 "reg_ref": "previous",
                 "overwrite": False},
    
    "run_blur": True,
    "blur": {"kernel": (15,15),
             "sigma": 5,
             "img_fold_src": "",
             "overwrite": False},

    "run_cellpose": True,
    "cellpose": {"channel_to_seg":"GFP",
                 "model_type": "cyto3",
                 "diameter": 60.0,
                 "flow_threshold": 0.4,
                 "cellprob_threshold": 0.0,
                 "overwrite": True,
                 "img_fold_src": "",
                 "process_as_2D": False,
                 "save_as_npy": False,
                 "nuclear_marker": "",},
    
    "run_threshold": True,
    "threshold": {"channel_to_seg":"GFP",
                  "overwrite": False,
                  "manual_threshold": None,
                  "img_fold_src": "",},
}