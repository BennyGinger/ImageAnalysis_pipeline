settings = {
    "input_folder": '/home/ben/Dropbox/SOTE_Lab/Pyhton/Pipelines/Test_images/nd2/Run2',
    "init":{"active_channel_list": ['GFP','RFP'],'full_channel_list':[],"overwrite": True},
    
    "run_bg_sub": True,
    "run_chan_shift": False,
    "run_register": True,
    "run_blur": False,
    
    "bg_sub": {"sigma": 0.0,"size": 7,"overwrite": False},
    "chan_shift": {"reg_channel": "RFP","reg_mtd": "rigid_body","overwrite": False},
    "register": {"reg_channel": "RFP","reg_mtd": "rigid_body","reg_ref": "previous","overwrite": False},
    "blur": {"kernel": (15,15),"sigma": 5,"img_fold_src": None,"overwrite": False},

    "run_cellpose": True,
    "run_threshold": False,
    
    "cellpose": {"channel_to_seg":"GFP",
                 "model_type": "cyto2",
                 "nuclear_marker": "",
                 "overwrite": False,
                 "diameter": 100.0,"flow_threshold": 0.4,"cellprob_threshold": 0.0,"cellprob_radius": 50.0,"net_avg": False,"mask": True,"img_fold_src": None,},
}