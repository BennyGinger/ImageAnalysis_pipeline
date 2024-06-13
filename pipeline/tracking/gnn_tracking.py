from __future__ import annotations
from os import PathLike
from pathlib import Path
from pipeline.utilities.Experiment_Classes import Experiment
from pipeline.utilities.data_utility import track_mask_lst_src, load_stack, save_tif, get_img_prop
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


################################## Main function ##################################

def gnn_tracking(exp_path: PathLike, channel_to_track: str, model: str, max_travel_dist: int, img_fold_src: str, mask_fold_src: str, overwrite: bool=False, decision_threshold: float=0.5, manual_correct: bool=False, trim_incomplete_tracks: bool=False,**kwargs)-> None:
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
    
    # Set exp paths
    exp_path = Path(exp_path)
    save_path: Path = exp_path.joinpath('Masks_GNN_Track')
    save_path.mkdir(exist_ok=True)
    
    
    # Already processed?
    if any(file.match(f"*{channel_to_track}*") for file in save_path.glob('*.tif')) and not overwrite:
            # Log
        print(f" --> Cells have already been tracked for the '{channel_to_track}' channel")
        return

    # Track images
    print(f" --> Tracking cells for the '{channel_to_track}' channel")
    
    # Prepare tracking
    csvs_folder = exp_path.joinpath('gnn_files')
    csvs_folder.mkdir(exist_ok=True)
    seg_fold_src = exp_path.joinpath(mask_fold_src)
    img_fold_src: Path = exp_path.joinpath(img_fold_src)
    frames, n_slices = get_img_prop(list(img_fold_src.glob('*.tif')))
    model_dict = model_select(model=model)
    metadata = unpack_kwargs(kwargs)

    # Create csv files
    if n_slices==1: # check of 2D or 3D
        is_3d = False
        preprocess_seq2graph_clean.create_csv(input_images=img_fold_src, input_seg=seg_fold_src, input_model=model_dict['model_metric'], channel=channel_to_track, output_csv=csvs_folder)
    else:
        is_3d = True
        preprocess_seq2graph_3d.create_csv(input_images=img_fold_src, input_seg=seg_fold_src, input_model=model_dict['model_metric'], channel=channel_to_track, output_csv=csvs_folder)
    
    # Run the model    
    predict(ckpt_path=model_dict['model_lightning'], path_csv_output=csvs_folder, num_seq='01')

    # Postprocess the model output
    pp = Postprocess(is_3d=is_3d, type_masks='tif', merge_operation='AND', decision_threshold=decision_threshold,
                    path_inference_output=csvs_folder, directed=True, path_seg_result=seg_fold_src, max_travel_dist=max_travel_dist)
    
    pp.create_trajectory() # Several output available that are also saved in the class, if needed one day
    # np.savetxt("/home/Fabian/ImageData/all_frames_traject.csv", all_frames_traject, delimiter=",")
    # np.savetxt("/home/Fabian/ImageData/trajectory_same_label.csv", trajectory_same_label, delimiter=",")

    pp.fill_mask_labels(debug=False)
    
    #relabel the masks from ID 1 until n and add metadata
    relabel_masks(frames,list(save_path.glob('*.tif')),channel_to_track,metadata,trim_incomplete_tracks)
    
    if manual_correct: #write mdf file to manual correct the tracks later
        prepare_manual_correct(frames,list(save_path.glob('*.tif')),channel_to_track,exp_path)


################################## Helper functions ##################################
def model_select(model):
    if model not in MODEL:
        raise AttributeError(f"{model =} is not a valid modelname.")
    
    model_dict = {"model_metric": f"/home/ImageAnalysis_pipeline/pipeline/tracking/gnn_track/models/{model}/all_params.pth",
                  "model_lightning": f"/home/ImageAnalysis_pipeline/pipeline/tracking/gnn_track/models/{model}/{MODEL[model]}"}
    
    return model_dict

def create_mdf_file(exp_path: Path, points_df, channel_seg):
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
    mdf_filename = exp_path.joinpath(channel_seg+'.mdf')
    file = open(mdf_filename, "w")
    file.writelines(mdf_lst)
    file.close()
    print(f'--> .mdf trackingfile saved for the {channel_seg} channel')

def prepare_manual_correct(frames, mask_src_list, channel_seg, exp_path: Path):
    # Load masks
    mask_stack = load_stack(mask_src_list,[channel_seg],range(frames))
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
    create_mdf_file(exp_path, points_df, channel_seg)

def relabel_masks(frames, mask_src_list, channel_seg, metadata, trim_incomplete_tracks=False):
    # Load masks
    mask_stack = load_stack(mask_src_list,[channel_seg],range(frames))
    # trim incomplete tracks
    if trim_incomplete_tracks:
        trim_incomplete_track(mask_stack)
    #relabel the masks
    mask_stack, _, _ = relabel_sequential(mask_stack)
    #save the masks back into the folder, this time with metadata
    for frame, mask_path in enumerate(mask_src_list):
        save_tif(array=mask_stack[frame], save_path=mask_path,**metadata)

def unpack_kwargs(kwargs: dict)-> dict:
    """Function to unpack the kwargs and extract necessary variable."""
    if not kwargs:
        return {'finterval':None, 'um_per_pixel':None}
    
    # Unpack the kwargs
    metadata = {}
    for k,v in kwargs.items():
        if k in ['um_per_pixel','finterval']:
            metadata[k] = v
    
    # if kwargs did not contain metadata, set to None
    if not metadata:
        metadata = {'finterval':None, 'um_per_pixel':None}
    
    return metadata


if __name__ == "__main__":
    from time import time
    
    input_folder = '/home/Test_images/nd2/Run4'
    gnn_track = {"channel_to_track":"RFP",
                   "model":"PhC-C2DH-U373", #neutrophil, Fluo-N2DH-SIM+, Fluo-N2DL-HeLa, Fluo-N3DH-SIM+ (implement from server first!), PhC-C2DH-U373
                   'decision_threshold': 0.4, #between 0-1, 1=more interrupted tracks, 0= more tracks gets connected, checks for the confidence of the model for the connection of two cells
                   'max_travel_dist':50,
                   "mask_appear":5, # not implemented yet
                   "manual_correct":False,
                   "trim_incomplete_tracks":True,
                   "overwrite":True}
    
    
    exp_path = '/home/Test_images/nd2/Run4/c4z1t91v1_s1'
    start = time()
    gnn_tracking(exp_path=exp_path,
                 channel_to_track='RFP', 
                 model="PhC-C2DH-U373",
                 max_travel_dist=10,
                 img_fold_src="Images_Registered",
                 mask_fold_src="Masks_Cellpose",
                 overwrite=True,
                 decision_threshold=0.4,
                 manual_correct=False,
                 trim_incomplete_tracks=False)
    end = time()
    print(f"Time to process: {round(end-start,ndigits=3)} sec\n")