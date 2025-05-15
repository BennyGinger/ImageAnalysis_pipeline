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
    
    "blur": (False,
            {"sigma": 5,
            "overwrite": False}),

    "cellpose": (True,
                {"channel_to_seg":"RFP", 
                "model_type": "cyto2", #cyto2_cp3, cyto3, /home/Fabian/Models/Cellpose/twoFishMacrophage
                "diameter": 15,
                "flow_threshold": 0.7,
                "cellprob_threshold":0,
                "process_as_2D": True,
                "overwrite": False,}),
    
    "threshold": (False,
                {"channel_to_seg":"RFP",
                "manual_threshold": None,
                "img_fold_src": "",
                "overwrite": False,}),
    
    "iou_track": (False,
                  {"channel_to_track":"RFP", 
                   "stitch_thres_percent": 0.5,
                   "shape_thres_percent": 0.95,
                   "mask_appear":5,
                   "copy_first_to_start": True, 
                   "copy_last_to_end": True,
                   "overwrite":False}),
    
    "gnn_track": (True,                         #not working: Fluo-C2DL-Huh7
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

# settings = {
    
#     # MUST !!!!!! Folder containing all the exp to be analyzed.
#     "input_folder": '/home/New_test',
    
#     # Channels are optional, if not given the pipeline will generate generic name: C1, C2...
#     "init":{"active_channel_list": ['GFP','RFP'], # name or list[name], channel(s) you want to analyze (order doesn't matter)
#             'full_channel_list':['GFP','RFP'],    # name or list[name], all channels present in the image file (ORDER MATTER)
#             "overwrite": False},                  # If set True, will overwrite all subsequent analysis
    
#     #######################################################
#     # Fromat: 
#     # "process name" : (True/False,                 whether you want to run or not that process
#     #                   {settings of process})      settings of that process
#     #######################################################
    
#     # Preprocess settings
#     "bg_sub": (True,
#                 {"overwrite": False}),
    
#     "chan_shift": (True,                          # Correct for shift between channels
#                     {"reg_channel": "RFP",        # Reference channel
#                     "reg_mtd": "rigid_body",      # Mtd: 'translation', 'rigid_body', 'scaled_rotation', 'affine','bilinear'
#                     "overwrite": False}),
    
#     "frame_shift": (True,                         # Correct for shift between frames (all channels)
#                 {"reg_channel": "RFP",            # Reference channel
#                 "reg_mtd": "rigid_body",          # Mtd: 'translation', 'rigid_body', 'scaled_rotation', 'affine','bilinear'
#                 "img_ref": "previous",            # Reference image: "first", "previous", "mean"
#                 "overwrite": False}),
    
#     "blur": (False,                               # Blur the image
#             {"sigma": 5,                          # 0-100, increased sigma, increases blur
#             "overwrite": False}),

#     "cellpose": (True,
#                 {"channel_to_seg":"RFP",          # name or list[name], Channel to segment
#                 "model_type": "cyto2",            # model to use: "cyto2", 'cyto2_cp3', 'cyto3', '/home/Fabian/Models/Cellpose/twoFishMacrophage'
#                 "diameter": 50.0,                 # Average diameter of cells (pixel)
#                 "flow_threshold": 0.3,            # Number and shape of mask generated from [0-2]: 0 = strict: low number of masks with uniform shape; 2 = loose: high number of masks with no coherent shape
#                 "cellprob_threshold":0,           # Probablity of a mask beeing a real cell [-6:6]: -6 = low: return all generated masks; 6 = high: return only high probability masks
#                 "process_as_2D": True,            # If true, will process 3D (with z-stack) as MaxIP
#                 "overwrite": False,}),
    
#     "threshold": (False,                          # Simple segmentation based on intensity threshold
#                 {"channel_to_seg":"RFP",          # name or list[name], Channel to segment
#                 "manual_threshold": None,         # int: Impose an intensity threshold, if set to None, then performs an automatic threshold
#                 "overwrite": False,}),
    
#     "iou_track": (True,                           # Static tracking algorythm for non migrating/non moving cells, e.g. cell culture
#                   {"channel_to_track":"RFP",      # name or list[name], Channel to segment
#                    "stitch_thres_percent": 0.6,   # 0-1, Percentage overlap between cells of different frames. 0: low overlap, will produce loads of false positive tracks; 1: high overlap, will remove many positive tracks
#                    "shape_thres_percent": 0.95,   # 0-1, Percentage of shape similarity between 2 frames. 0: low similarity between cells, will keep merging/dividing segmentation; 1: high similarity, will remove mask that are different
#                    "mask_appear":5,               # Number of masks present in the track to be considered as true cells. (Remove floating cells)
#                    "copy_first_to_start": True,   # if uncomplete track, copy the first mask to the start of the frame sequence
#                    "copy_last_to_end": True,      # if uncomplete track, copy the last mask to the end of the frame sequence
#                    "overwrite":False}),
    
#     "gnn_track": (False,                          # Dynamic tracking algorythm for migrating cells, e.g. fish migration
#                   {"channel_to_track":"RFP",      # name or list[name], Channel to segment TODO: check if list can go and check if channel exists
#                    "model":"PhC-C2DH-U373",       # neutrophil, Fluo-N2DH-SIM+, Fluo-N2DL-HeLa, Fluo-N3DH-SIM+ (implement from server first!), PhC-C2DH-U373
#                    'decision_threshold': 0.4,     # between 0-1, 1=more interrupted tracks, 0= more tracks gets connected, checks for the confidence of the model for the connection of two cells
#                    'max_travel_dist':50,          # Maximum distance a cell can travel between frame (pixel)
#                    "manual_correct":True,         # Save output as md file to be able to modify the tracks with TrackMJ in ImageJ
#                    "overwrite":False}),
    
#     "man_track": (False,                          # Requires csv file with "channel.csv" format in the exp folder
#                   {"channel_to_track":"BF",       # name or list[name], Channel to segment TODO: check if list can go and check if channel exists
#                    "track_seg_mask": False,       # If True, will overlap the tracking with segmented masks and change the mask value accordingly
#                    "radius": 5,                   # Radius of the mask to be drawn around the track
#                    'copy_first_to_start': True,
#                    'copy_last_to_end': True,
#                    'mask_appear': 2,
#                    "dilate_value":20,             # Only if track_seg_mask == True. Value of dilation of the mask
#                    "process_as_2D":True,
#                    "overwrite":False}),
    
#     "draw_mask": (True, 
#                   {"mask_label": "wound",         # name or list[name]
#                    "channel_show": "RFP",         # Channel to display during drawing (no impact on the analysis)
#                    "overwrite": False}),
    
#     "extract_data": (True,
#                   {"overwrite": True}),
# }