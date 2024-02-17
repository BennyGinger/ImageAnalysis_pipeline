from __future__ import annotations
from os import getcwd
import sys
parent_dir = getcwd()
sys.path.append(parent_dir)
from time import time
# TODO: fix module path 
from ImageAnalysis_pipeline.pipeline.Experiment_Classes import Experiment
from ImageAnalysis_pipeline.pipeline.pre_process.main_pre_process import pre_process_all
from ImageAnalysis_pipeline.pipeline.segmentation.segmentation import threshold
from ImageAnalysis_pipeline.pipeline.segmentation.cp_segmentation import cellpose_segmentation
from ImageAnalysis_pipeline.pipeline.tracking.iou_tracking import iou_tracking
from ImageAnalysis_pipeline.pipeline.analysis.channel_data import extract_channel_data

def change_attribute(exp_set_list: list[Experiment], attribute: str, value: any)-> list[Experiment]:
    for exp_set in exp_set_list:
        exp_set.set_attribute(attribute,value)
        exp_set.save_as_json()
    return exp_set_list

if __name__ == "__main__":

    t1 = time()
    
    parent_folder = '/home/Fabian/ImageData'
    preprocess_parameters = {'parent_folder': parent_folder,
                             'active_channel_list': ['GFP','RFP'],
                             'full_channel_list':['GFP','RFP'], 
                             'file_type': '.nd2',
                             'img_seq_overwrite': False,
                             'bg_sub': False,'sigma': 0.0,'size': 7,'bg_sub_overwrite': False,
                             'chan_shift': False, 'reg_channel': 'RFP', 'reg_mtd': 'rigid_body', 'chan_shift_overwrite': False,
                             'register_images': True, 'reg_ref': 'previous', 'reg_overwrite': False,
                             'blur': False, 'blur_kernel': (15,15), 'blur_sigma': 5,'img_fold_src': None, 'blur_overwrite': False,}
    
    segmentation_parameters = {'channel_seg':'GFP','manual_threshold': 75, 'thresold_overwrite': True, 'img_fold_src': 'Images_Registered'}
    
    cp_segmentation_parameters = {'channel_seg':'GFP','model_type':'/home/Fabian/Models/Cellpose/twoFishMacrophage','nuclear_marker':None,'as_2D':True,
                                  'cellpose_overwrite':False,'stitch':None,'img_fold_src':'Images_Registered',
                                  'diameter':13,'flow_threshold':0,'cellprob_threshold':-1.5,'gpu':True,"as_npy":False,}
    
    iou_tracking_parameters = {'channel_seg':'RFP','mask_fold_src':'Masks_Cellpose','stitch_thres_percent':0.75,
                               'shape_thres_percent':0.1,'iou_track_overwrite':False, 'n_mask': 10}
    
    # if cp_segmentation_parameters['cellpose_overwrite']:
    #     iou_tracking_parameters['iou_track_overwrite'] = True
    
    exp_set_list = pre_process_all(**preprocess_parameters)
    
    # exp_set_list = threshold(exp_set_list,**segmentation_parameters)
    exp_set_list = cellpose_segmentation(exp_set_list,**cp_segmentation_parameters)
    # exp_set_list = iou_tracking(exp_set_list,**iou_tracking_parameters)
    print(exp_set_list[0])
    # Add interval_sec manually
    # exp_set_list = change_attribute(exp_set_list,'interval_sec',10)
    
    # exp_set_list = extract_channel_data(exp_set_list,'Images_Registered',False)
    
    t2 = time()
    if t2-t1<60: print(f"Time to process: {round(t2-t1,ndigits=3)} sec\n")
    else: print(f"Time to process: {round((t2-t1)/60,ndigits=1)} min\n")