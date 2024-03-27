from __future__ import annotations
from image_handeling.Experiment_Classes import Experiment
from image_handeling.data_utility import is_processed, create_save_folder, delete_old_masks
from tracking.gnn_track.inference_clean import predict
from tracking.gnn_track.postprocess_clean import Postprocess
from tracking.gnn_track import preprocess_seq2graph_clean, preprocess_seq2graph_3d
from os.path import join

def model_select(model):
    model_dict = {}
    if model == 'neutrophil':
        model_dict['model_metric'] = './models/neutrophil/all_params.pth'
        model_dict['model_lightning'] = './models/neutrophil/epoch=73.ckpt'   
    else:
        raise AttributeError(f'{model =} is not a valid modelname.')
    return model_dict


# # # # # # # # main functions # # # # # # # # # 

def gnn_tracking(exp_set_list: list[Experiment], channel_seg: str, model:str, gnn_track_overwrite: bool=False, img_fold_src: str = None, mask_fold_src: str = None, morph: bool=False, n_mask=2, min_cell_size:int = 20, decision_threshold:float = 0.5):
    for exp_set in exp_set_list:
        # Activate the branch
        exp_set.masks.is_gnn_tracking = True
        # Check if exist
        if is_processed(exp_set.masks.manual_tracking,channel_seg,gnn_track_overwrite):
                # Log
            print(f" --> Cells have already been tracked for the '{channel_seg}' channel")
            continue

        # Track images
        print(f" --> Tracking cells for the '{channel_seg}' channel")

        # Create save folder and remove old masks
        #create_save_folder(exp_set.exp_path,'Masks_GNN_Track')
        create_save_folder(exp_set.exp_path,'gnn_files')
        files_folder = join(exp_set.exp_path,'gnn_files')
        create_save_folder(files_folder,'track_csv')
        
        delete_old_masks(exp_set.masks.manual_tracking,channel_seg,exp_set.man_tracked_masks_lst,gnn_track_overwrite)
        
        model_dict = model_select(model=model)
        input_img = join(exp_set.exp_path, img_fold_src)
        input_seg = join(exp_set.exp_path, mask_fold_src)
        csv_path = join(files_folder, 'track_csv')

        if len(exp_set.img_properties.n_slices)==1: # check of 2D or 3D
            is_3d = False
            preprocess_seq2graph_clean.create_csv(input_images=input_img, input_seg=input_seg, input_model=model_dict['model_metric'], channel=channel_seg, output_csv=csv_path, min_cell_size=min_cell_size)
        else:
            is_3d = True
            preprocess_seq2graph_3d.create_csv(input_images=input_img, input_seg=input_seg, input_model=model_dict['model_metric'], channel=channel_seg, output_csv=csv_path, min_cell_size=min_cell_size)
            
        predict(ckpt_path=model_dict['model_lightning'], path_csv_output=csv_path, num_seq='01')
                
        pp = Postprocess(is_3d=is_3d, type_masks='tif', merge_operation='AND', decision_threshold=decision_threshold,
                     path_inference_output=files_folder, center_coord=False, directed=True, path_seg_result=input_seg)
        
        pp.create_trajectory() # Several output available that are also saved in the class, if needed one day
        pp.fill_mask_labels(debug=False)
    
    # Save settings
    exp_set.masks.gnn_tracking[channel_seg] = {'img_fold_src':img_fold_src, 'mask_fold_src':mask_fold_src, 'model':model, 'n_mask':n_mask, 'morph':morph, 'min_cell_size':min_cell_size, 'decision_threshold':decision_threshold}
    exp_set.save_as_json()


