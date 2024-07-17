from __future__ import annotations
from pathlib import Path
from pipeline.utilities.data_utility import load_stack, save_tif, get_exp_props
from pipeline.utilities.pipeline_utility import PathType
from pipeline.tracking.gnn_track.prediction.prediction import predict
from pipeline.tracking.gnn_track.postprocess.postprocess_clean import Postprocess
from pipeline.tracking.gnn_track.feature_extraction.feature_extraction import extract_img_features
from pipeline.mask_transformation.complete_track import trim_incomplete_track
from skimage.segmentation import relabel_sequential
from skimage.measure import regionprops_table
from matplotlib.colors import cnames
from pandas import DataFrame
import json
from time import time

import numpy as np #TODO remove later!

# List of all the models. For organization purposes, the models are divided into two categories: BUILD_IN_MODEL and IN_HOUSE_MODEL. However, they are then combined into a single dictionary (MODEL) for ease of use. The key is the model name and the value is the name of the checkpoint file to use.
BUILD_IN_MODEL = {"Fluo-C2DL-Huh7": "epoch=136.ckpt", "Fluo-N2DH-SIM+": "epoch=132.ckpt", "Fluo-N3DH-SIM+": "epoch=42.ckpt", "Fluo-N2DL-HeLa": "epoch=312.ckpt", "PhC-C2DH-U373": "epoch=10.ckpt"}
IN_HOUSE_MODEL = {"neutrophil_old": "epoch=73.ckpt", "neutrophil": "epoch=175.ckpt"}
MODEL = {**BUILD_IN_MODEL, **IN_HOUSE_MODEL}

# [ ] Need to test the 3D tracking
################################## Main function ##################################

def gnn_tracking(exp_path: PathType, channel_to_track: str, model: str, max_travel_dist: int, img_fold_src: str, seg_fold_src: str, overwrite: bool=False, decision_threshold: float=0.5, manual_correct: bool=False, trim_incomplete_tracks: bool=False, directed: bool=False, **kwargs)-> None:
    """
    Perform GNN Tracking based cell tracking on a list of experiments.

    Args:
        exp_obj_lst (list[Experiment]): List of Experiment objects to perform tracking on.
        channel_seg (str): Channel name for segmentation.
        model (str): name of the model (at the moment only 'neutrophil' or 'neutrophil_old')
        overwrite (bool, optional): Flag to overwrite existing tracking results. Defaults to False.
        img_fold_src (str): Source folder name for images.
        mask_fold_src (str): Source folder name for masks.
        morph: bool=False (not included yet)
        mask_appear (int, optional): Number of times a mask should appear to be considered valid. Defaults to 2.
        decision_threshold (float, optional): #0 to 1, 1=more interrupted tracks, 0= more tracks gets connected.(Source: ChatGPT) Defaults to 0.5.
        manual_correct (bool, optional): Flag to create .mdf file for ImageJ plugin MTrackJ to manual correct the tracks. Defaults to False.
    
    Returns:
        list[Experiment]: List of Experiment objects with updated tracking information.
    """    
    # Get the arguments to store them in file and assess if some part of the function needs to be overwritten or not
    passed_args = locals()
    passed_args.update(kwargs)
    
    # Set exp paths
    exp_path: Path = Path(exp_path)
    save_path: Path = exp_path.joinpath('Masks_GNN_Track')
    save_path.mkdir(exist_ok=True)
    
    
    # Already processed?
    if any(file.match(f"*{channel_to_track}*") for file in save_path.glob('*.tif')) and not overwrite:
            # Log
        print(f" --> Cells have already been tracked for the '{channel_to_track}' channel")
        return

    # Track images
    print(f" --> Tracking cells for the '{channel_to_track}' channel")
    
    ## Prepare tracking
    # Set all the paths
    img_paths, preds_dir, seg_paths, model_path, ckpt_path = set_all_paths(exp_path, model, img_fold_src, seg_fold_src)
    # Get the properties of the experiment
    _, _, frames, z_slices = get_exp_props(list(img_paths.glob('*.tif')))
    is_3d = True if z_slices > 1 else False

    # Create csv files
    ow_extract_feat = overwrite_extraction_feat(passed_args,preds_dir)
    extract_img_features(img_paths=img_paths,
                         seg_paths=seg_paths,
                         model_path=model_path,
                         save_dir=preds_dir,
                         channel=channel_to_track,
                         overwrite=ow_extract_feat)
    
    # Run the model    
    start = time()
    predict(ckpt_path=ckpt_path,
            prediction_dir=preds_dir,
            is_3d=is_3d,
            max_travel_pix=max_travel_dist, 
            directed=directed)
    end = time()
    print(f"Time to predict: {round(end-start,ndigits=3)} sec\n")
    
    # Postprocess the model output
    start = time()
    pp = Postprocess(is_3d=is_3d,
                     seg_paths=seg_paths,
                     preds_dir=preds_dir,
                     decision_threshold=decision_threshold,
                     merge_operation='AND',
                     max_travel_dist=max_travel_dist,
                     directed=directed,
                     channel_to_track=channel_to_track)
    
    all_frames_traject, trajectory_same_label = pp.create_trajectory() # Several output available that are also saved in the class, if needed one day
    all_frames_path = preds_dir.joinpath(f'all_frames_traject.csv')
    traj_path = preds_dir.joinpath(f'trajectory_same_label.csv')
    np.savetxt(all_frames_path, all_frames_traject, delimiter=",")
    np.savetxt(traj_path, trajectory_same_label, delimiter=",")

    pp.fill_mask_labels(save_path=save_path)
    end = time()
    print(f"Time to postprocess: {round(end-start,ndigits=3)} sec\n")
    
    # Relabel the masks from ID 1 until n and add metadata
    metadata = unpack_kwargs(kwargs)
    relabel_masks(frames,save_path,channel_to_track,metadata,trim_incomplete_tracks)
    
    if manual_correct: #write mdf file to manual correct the tracks later
        prepare_manual_correct(frames,save_path,channel_to_track,exp_path)




################################## Helper functions ##################################
def set_all_paths(exp_path: Path, model: str, img_fold_src: Path, seg_fold_src: Path)-> tuple[Path, Path, Path, Path, Path]:
    # Create save directory
    preds_dir: Path = exp_path.joinpath('gnn_files')
    preds_dir.mkdir(exist_ok=True)
    # Get paths for the images and seg masks
    seg_paths = exp_path.joinpath(seg_fold_src)
    img_paths: Path = exp_path.joinpath(img_fold_src)
    # Get the model paths
    if model not in MODEL:
        raise AttributeError(f"{model =} is not a valid modelname.")
    model_path = Path(f"/home/ImageAnalysis_pipeline/pipeline/tracking/gnn_track/models/{model}/all_params.pth")
    ckpt_path = Path(f"/home/ImageAnalysis_pipeline/pipeline/tracking/gnn_track/models/{model}/{MODEL[model]}")
    return img_paths,preds_dir,seg_paths,model_path,ckpt_path

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

def prepare_manual_correct(frames: int, mask_fold_src: Path, channel_seg: str, exp_path: Path):
    # Load masks
    # Load masks
    mask_src_list = sorted(list(mask_fold_src.glob('*.tif')))
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

def relabel_masks(frames: int, mask_fold_src: Path, channel_seg: str, metadata: dict, trim_incomplete_tracks: bool=False):
    # Load masks
    mask_fold_src = Path(mask_fold_src)
    mask_src_list = sorted(list(mask_fold_src.glob('*.tif')))
    mask_stack = load_stack(mask_src_list,[channel_seg],range(frames))
    # trim incomplete tracks
    print(f"Unique mask before trim: {len(np.unique(mask_stack))}")
    if trim_incomplete_tracks:
        trim_incomplete_track(mask_stack)
        print(f"Unique mask after trim: {len(np.unique(mask_stack))}")
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

def overwrite_extraction_feat(local_args: dict, save_dir: Path)-> bool:
    """Function to check if the extraction features should be overwritten based on the arguments passed."""
    if 'ow_extract_feat' in local_args['kwargs']:
        return local_args['kwargs']['ow_extract_feat']
    
    def write_json(args: dict, path: Path)-> None:
        with open(path,'w') as file:
            json.dump(args,file)
    
    args_path = save_dir.joinpath('extraction_feat.json')
    if not args_path.exists():
        write_json(local_args,args_path)
        return True
    
    with open(args_path,'r') as file:
        args_dict = json.load(file)
    
    # Check if any of those key/value pairs are different
    key_ref = ['img_fold_src','seg_fold_src','model']
    for key in key_ref:
        if args_dict[key] != local_args[key]:
            write_json(local_args,args_path)
            return True
    return False

if __name__ == "__main__":
    from time import time
    
    exp_path = '/home/Test_images/CTC_Dataset/PhC-C2DH-U373/U373_1_s1'
    start = time()
    gnn_tracking(exp_path=exp_path,
                 channel_to_track='BF', 
                 model="PhC-C2DH-U373",
                 max_travel_dist=15,
                 img_fold_src="Images",
                 seg_fold_src="Masks_Cellpose",
                 overwrite=True,
                 decision_threshold=0.4,
                 manual_correct=False,
                 trim_incomplete_tracks=False,
                 directed=True,
                )
    end = time()
    print(f"Time to process: {round(end-start,ndigits=3)} sec\n")