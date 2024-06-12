from __future__ import annotations
from pipeline.utilities.Experiment_Classes import Experiment
from pipeline.utilities.data_utility import is_processed, create_save_folder, delete_old_masks, seg_mask_lst_src, img_list_src, track_mask_lst_src, load_stack, save_tif
from pipeline.tracking.gnn_track.inference_clean import predict
from pipeline.tracking.gnn_track.postprocess_clean import Postprocess
from pipeline.tracking.gnn_track import preprocess_seq2graph_clean, preprocess_seq2graph_3d
from pipeline.mask_transformation.complete_track import trim_incomplete_track
from skimage.segmentation import relabel_sequential
from skimage.measure import regionprops_table
from matplotlib.colors import cnames
from pandas import DataFrame
from os.path import join

import numpy as np #TODO remove later!

# List of all the models. For organization purposes, the models are divided into two categories: BUILD_IN_MODEL and IN_HOUSE_MODEL. However, they are then combined into a single dictionary (MODEL) for ease of use. The key is the model name and the value is the name of the checkpoint file to use.
BUILD_IN_MODEL = {"Fluo-C2DL-Huh7": "epoch=136.ckpt", "Fluo-N2DH-SIM+": "epoch=132.ckpt", "Fluo-N3DH-SIM+": "epoch=42.ckpt", "Fluo-N2DL-HeLa": "epoch=312.ckpt", "PhC-C2DH-U373": "epoch=10.ckpt"}
IN_HOUSE_MODEL = {"neutrophil_old": "epoch=73.ckpt", "neutrophil": "epoch=175.ckpt"}
MODEL = {**BUILD_IN_MODEL, **IN_HOUSE_MODEL}

def model_select(model):
    if model not in MODEL:
        raise AttributeError(f"{model =} is not a valid modelname.")
    
    model_dict = {"model_metric": f"/ImageAnalysis/pipeline/tracking/gnn_track/models/{model}/all_params.pth",
                  "model_lightning": f"/ImageAnalysis/pipeline/tracking/gnn_track/models/{model}/{MODEL[model]}"}
    
    return model_dict

def create_mdf_file(exp_obj, points_df, channel_seg):
    mdf_lst = []
    mdf_lst.append('MTrackJ 1.5.1 Data File\n')
    mdf_lst.append('Displaying true true true 1 2 0 3 100 4 0 0 0 2 1 12 0 true true true false\n')
    mdf_lst.append('Assembly 1 FF0000\n')
    mdf_lst.append('Cluster 1 FF0000\n')
    colour_ID = 0
    hex_colour_lst = list(cnames.values())
    for track_ID, row in points_df.iterrows():
        if colour_ID >= len(hex_colour_lst):
            colour_ID = 0
        mdf_lst.append(f'Track {track_ID} {hex_colour_lst[colour_ID]} true\n')
        colour_ID += 1
        point_ID = 0
        for frame, centroid in enumerate(row, start=1):
            if not centroid != centroid:
                point_ID += 1
                mdf_lst.append(f'Point {point_ID} {centroid[1]} {centroid[0]} 1.0 {float(frame)} 1.0\n')
    mdf_lst.append('End of MTrackJ Data File\n')
    mdf_filename = join(exp_obj.exp_path, channel_seg+'.mdf')
    file = open(mdf_filename, "w")
    file.writelines(mdf_lst)
    file.close()
    print(f'--> .mdf trackingfile saved for the {channel_seg} channel')

def prepare_manual_correct(exp_obj, channel_seg, mask_fold_src):
    # Load masks
    mask_src_list = track_mask_lst_src(exp_obj,mask_fold_src)
    mask_stack = load_stack(mask_src_list,[channel_seg],range(exp_obj.img_properties.n_frames))
    # get centroids of all obj and save them with the label ID in a dataframe
    for frame, img in enumerate(mask_stack, start=1):
        props_table = regionprops_table(img, properties=('label','centroid'))
        props_df = DataFrame(props_table)
        props_df[frame] = list(zip(props_df['centroid-0'], props_df['centroid-1']))
        props_df = props_df.drop(columns=['centroid-0', 'centroid-1'])
        if frame == 1:
            points_df=props_df[['label', frame]].copy()
        else:
            points_df = points_df.join(props_df.set_index('label'),on='label', how='outer')
    points_df = points_df.sort_values('label').set_index('label')
    create_mdf_file(exp_obj, points_df, channel_seg)

def relabel_masks(exp_obj, channel_seg, mask_fold_src, trim_incomplete_tracks=False):
    # Load masks
    mask_src_list = track_mask_lst_src(exp_obj,mask_fold_src)
    mask_stack = load_stack(mask_src_list,[channel_seg],range(exp_obj.img_properties.n_frames))
    # trim incomplete tracks
    if trim_incomplete_tracks:
        trim_incomplete_track(mask_stack)
    #relabel the masks
    mask_stack, _, _ = relabel_sequential(mask_stack)
    #save the masks back into the folder, this time with metadata
    for frame, mask_path in enumerate(mask_src_list):
        save_tif(array=mask_stack[frame], save_path=mask_path, um_per_pixel=exp_obj.analysis.um_per_pixel, finterval=exp_obj.analysis.interval_sec)

# # # # # # # # main functions # # # # # # # # # 

def gnn_tracking(exp_obj_lst: list[Experiment], channel_seg: str, model:str, max_travel_dist:int ,overwrite: bool=False,
                 img_fold_src: str = None, mask_fold_src: str = None, morph: bool=False, mask_appear=2,
                 decision_threshold:float = 0.5, manual_correct:bool=False, trim_incomplete_tracks:bool=False):
    """
    Perform GNN Tracking based cell tracking on a list of experiments.

    Args:
        exp_obj_lst (list[Experiment]): List of Experiment objects to perform tracking on.
        channel_seg (str): Channel name for segmentation.
        model (str): name of the model (at the moment only 'neutrophil' or 'neutrophil_old')
        overwrite (bool, optional): Flag to overwrite existing tracking results. Defaults to False.
        img_fold_src (str): Source folder path for images.
        mask_fold_src (str): Source folder path for masks.
        morph: bool=False (not included yet)
        mask_appear (int, optional): Number of times a mask should appear to be considered valid. Defaults to 2.
        decision_threshold (float, optional): #0 to 1, 1=more interrupted tracks, 0= more tracks gets connected.(Source: ChatGPT) Defaults to 0.5.
        manual_correct (bool, optional): Flag to create .mdf file for ImageJ plugin MTrackJ to manual correct the tracks. Defaults to False.
    
    Returns:
        list[Experiment]: List of Experiment objects with updated tracking information.
    """    
    for exp_obj in exp_obj_lst:
        # Activate the branch
        exp_obj.tracking.is_gnn_tracking = True

        
        # Already processed?
        if is_processed(exp_obj.tracking.gnn_tracking,channel_seg,overwrite):
                # Log
            print(f" --> Cells have already been tracked for the '{channel_seg}' channel")
            continue

        # Track images
        print(f" --> Tracking cells for the '{channel_seg}' channel")
        
        # Create save folder and remove old masks
        create_save_folder(exp_obj.exp_path,'Masks_GNN_Track')
        create_save_folder(exp_obj.exp_path,'gnn_files')
        files_folder = join(exp_obj.exp_path,'gnn_files')
        delete_old_masks(exp_obj.tracking.gnn_tracking,channel_seg,exp_obj.gnn_tracked_masks_lst,overwrite)
        model_dict = model_select(model=model)
        
        #get path of mask
        mask_fold_src, _ = seg_mask_lst_src(exp_obj,mask_fold_src)
        input_seg = join(exp_obj.exp_path, mask_fold_src)

        #get path of image
        img_fold_src, _ = img_list_src(exp_obj,img_fold_src)
        input_img = join(exp_obj.exp_path, img_fold_src)
        if exp_obj.img_properties.n_slices==1: # check of 2D or 3D
            is_3d = False
            preprocess_seq2graph_clean.create_csv(input_images=input_img, input_seg=input_seg, input_model=model_dict['model_metric'], channel=channel_seg, output_csv=files_folder)
        else:
            is_3d = True
            preprocess_seq2graph_3d.create_csv(input_images=input_img, input_seg=input_seg, input_model=model_dict['model_metric'], channel=channel_seg, output_csv=files_folder)
            
        predict(ckpt_path=model_dict['model_lightning'], path_csv_output=files_folder, num_seq='01')
   
        pp = Postprocess(is_3d=is_3d, type_masks='tif', merge_operation='AND', decision_threshold=decision_threshold,
                     path_inference_output=files_folder, directed=True, path_seg_result=input_seg, max_travel_dist=max_travel_dist)
        
        pp.create_trajectory() # Several output available that are also saved in the class, if needed one day
        # np.savetxt("/home/Fabian/ImageData/all_frames_traject.csv", all_frames_traject, delimiter=",")
        # np.savetxt("/home/Fabian/ImageData/trajectory_same_label.csv", trajectory_same_label, delimiter=",")


        pp.fill_mask_labels(debug=False)
        
        
        
        #relabel the masks from ID 1 until n and add metadata
        relabel_masks(exp_obj, channel_seg, mask_fold_src = 'Masks_GNN_Track',trim_incomplete_tracks=trim_incomplete_tracks)
        
        if manual_correct: #write mdf file to manual correct the tracks later
            prepare_manual_correct(exp_obj, channel_seg, mask_fold_src = 'Masks_GNN_Track')
            
        # Save settings
        exp_obj.tracking.gnn_tracking[channel_seg] = {'img_fold_src':img_fold_src, 'mask_fold_src':mask_fold_src, 'model':model, 'mask_appear':mask_appear, 'morph':morph, 'decision_threshold':decision_threshold, 'max_travel_dist':max_travel_dist}
        exp_obj.save_as_json()
    return exp_obj_lst


