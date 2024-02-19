from __future__ import annotations
from Experiment_Classes import Experiment
from loading_data import is_processed, create_save_folder, delete_old_masks
import preprocess_seq2graph_clean, inference_clean, postprocess_clean







def model_select(model):
    model_dict = {}
    if model == 'neutrophil':
        model_dict['model_metric'] = '/gnn_track/models/neutrophil/all_params.pth'
        model_dict['model_lightning'] = '/gnn_track/models/neutrophil/epoch=73.ckpt'   
    else:
        raise AttributeError(f'{model =} is not a valid modelname.')
    return model_dict




# # # # # # # # main functions # # # # # # # # # 

def gnn_tracking(exp_set_list: list[Experiment], channel_seg: str, model:str, gnn_track_overwrite: bool=False, img_fold_src: str = None, mask_fold_src: str = None, morph: bool=False, n_mask=2):
    for exp_set in exp_set_list:
        # Check if exist
        if is_processed(exp_set.masks.manual_tracking,channel_seg,gnn_track_overwrite):
                # Log
            print(f" --> Cells have already been tracked for the '{channel_seg}' channel")
            continue

        # Track images
        print(f" --> Tracking cells for the '{channel_seg}' channel")

        # Create save folder and remove old masks
        create_save_folder(exp_set.exp_path,'Masks_GNN_Track')
        delete_old_masks(exp_set.masks.manual_tracking,channel_seg,exp_set.mask_manual_track_list,gnn_track_overwrite)
        
        model_dict = model_select(model=model)
        

        if len(exp_set_list['img_properties']['n_slices'])==1: # check of 2D or 3D
            import preprocess_seq2graph_clean
            preprocess_seq2graph_clean.create_csv(input_images=input_images, input_seg=input_seg, input_model=model_dict['model_metric'], channel=channel_seg, output_csv=output_csv, min_cell_size=min_cell_size)
        else:
            import preprocess_seq2graph_3d
    
    
    
    # Save settings
    exp_set.masks.gnn_tracking[channel_seg] = {'img_fold_src':img_fold_src, 'mask_fold_src':mask_fold_src, 'model':model, 'n_mask':n_mask, 'morph':morph}
    exp_set.save_as_json()


