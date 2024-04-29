from __future__ import annotations
from image_handeling.Experiment_Classes import Experiment
from image_handeling.data_utility import is_processed, create_save_folder, delete_old_masks, seg_mask_lst_src, img_list_src
from tracking.gnn_track.inference_clean import predict
from tracking.gnn_track.postprocess_clean import Postprocess
from tracking.gnn_track import preprocess_seq2graph_clean, preprocess_seq2graph_3d
from os.path import join
from os import getcwd
from pathlib import Path



def model_select(model):
    model_dict = {}
    root = getcwd()
    if model == 'neutrophil_old':
        model_dict['model_metric'] = root+'/ImageAnalysis_pipeline/pipeline/tracking/gnn_track/models/neutrophil_old/all_params.pth'
        model_dict['model_lightning'] =  root+'/ImageAnalysis_pipeline/pipeline/tracking/gnn_track/models/neutrophil_old/epoch=73.ckpt'
    elif model == 'neutrophil':
        model_dict['model_metric'] =  root+'/ImageAnalysis_pipeline/pipeline/tracking/gnn_track/models/neutrophil/all_params.pth'
        model_dict['model_lightning'] =  root+'/ImageAnalysis_pipeline/pipeline/tracking/gnn_track/models/neutrophil/epoch=175.ckpt'
    elif model == 'Fluo-C2DL-Huh7':
        model_dict['model_metric'] =  root+'/ImageAnalysis_pipeline/pipeline/tracking/gnn_track/models/Fluo-C2DL-Huh7/all_params.pth'
        model_dict['model_lightning'] =  root+'/ImageAnalysis_pipeline/pipeline/tracking/gnn_track/models/Fluo-C2DL-Huh7/epoch=136.ckpt'
    elif model == 'Fluo-N2DH-SIM+':
        model_dict['model_metric'] =  root+'/ImageAnalysis_pipeline/pipeline/tracking/gnn_track/models/Fluo-N2DH-SIM+/all_params.pth'
        model_dict['model_lightning'] =  root+'/ImageAnalysis_pipeline/pipeline/tracking/gnn_track/models/Fluo-N2DH-SIM+/epoch=132.ckpt'
    elif model == 'Fluo-N2DL-HeLa':
        model_dict['model_metric'] =  root+'/ImageAnalysis_pipeline/pipeline/tracking/gnn_track/models/Fluo-N2DL-HeLa/all_params.pth'
        model_dict['model_lightning'] =  root+'/ImageAnalysis_pipeline/pipeline/tracking/gnn_track/models/Fluo-N2DL-HeLa/epoch=312.ckpt'
    elif model == 'Fluo-N3DH-SIM+':
        model_dict['model_metric'] =  root+'/ImageAnalysis_pipeline/pipeline/tracking/gnn_track/models/Fluo-N3DH-SIM+/all_params.pth'
        model_dict['model_lightning'] =  root+'/ImageAnalysis_pipeline/pipeline/tracking/gnn_track/models/Fluo-N3DH-SIM+/epoch=42.ckpt'
    elif model == 'PhC-C2DH-U373':
        model_dict['model_metric'] =  root+'/ImageAnalysis_pipeline/pipeline/tracking/gnn_track/models/PhC-C2DH-U373/all_params.pth'
        model_dict['model_lightning'] =  root+'/ImageAnalysis_pipeline/pipeline/tracking/gnn_track/models/PhC-C2DH-U373/epoch=10.ckpt'
    else:
        raise AttributeError(f'{model =} is not a valid modelname.')
    return model_dict


# # # # # # # # main functions # # # # # # # # # 

def gnn_tracking(exp_obj_lst: list[Experiment], channel_seg: str, model:str, overwrite: bool=False,
                 img_fold_src: str = None, mask_fold_src: str = None, morph: bool=False, mask_appear=2,
                 min_cell_size:int = 20, decision_threshold:float = 0.5, merge_operation:str='AND'):
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
        min_cell_size (int, optional): Minimum cell size to be recognized as cell. Defaults to 20.
        decision_threshold (float, optional): Not sure yet, what is does. Defaults to 0.5.
    

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
            preprocess_seq2graph_clean.create_csv(input_images=input_img, input_seg=input_seg, input_model=model_dict['model_metric'], channel=channel_seg, output_csv=files_folder, min_cell_size=min_cell_size)
        else:
            is_3d = True
            preprocess_seq2graph_3d.create_csv(input_images=input_img, input_seg=input_seg, input_model=model_dict['model_metric'], channel=channel_seg, output_csv=files_folder, min_cell_size=min_cell_size)
            
        predict(ckpt_path=model_dict['model_lightning'], path_csv_output=files_folder, num_seq='01')
                
        pp = Postprocess(is_3d=is_3d, type_masks='tif', merge_operation='AND', decision_threshold=decision_threshold,
                     path_inference_output=files_folder, center_coord=False, directed=True, path_seg_result=input_seg)
        
        pp.create_trajectory() # Several output available that are also saved in the class, if needed one day
        pp.fill_mask_labels(debug=False)
    
        # Save settings
        exp_obj.tracking.gnn_tracking[channel_seg] = {'img_fold_src':img_fold_src, 'mask_fold_src':mask_fold_src, 'model':model, 'mask_appear':mask_appear, 'morph':morph, 'min_cell_size':min_cell_size, 'decision_threshold':decision_threshold}
        exp_obj.save_as_json()
    return exp_obj_lst


