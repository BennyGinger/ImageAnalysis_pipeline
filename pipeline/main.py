from __future__ import annotations
# Force the multiprocessing to to start with new interpreter, as pytorch by default will pre-configure the interpreter, which will 'hog' CPU usage
from multiprocessing import set_start_method
set_start_method("spawn",force=True)
from time import time
import pandas as pd
from pipeline.image_extraction.ImageExtract_Module import ImageExtractionModule
from pipeline.pre_process.PreProcess_Module import PreProcessModule
from pipeline.segmentation.Segmentation_Module import SegmentationModule
from pipeline.tracking.Tracking_Class import TrackingModule
from pipeline.analysis.Analysis_class import AnalysisModule

def run_pipeline(settings: dict)-> pd.DataFrame:
    
    input_folder = settings['input_folder']
    exp_list = ImageExtractionModule(input_folder,**settings['init']).extract_img_seq()
    exp_list = PreProcessModule(input_folder,exp_list).process_from_settings(settings)
    exp_list = SegmentationModule(input_folder,exp_list).segment_from_settings(settings)
    exp_list = TrackingModule(input_folder,exp_list).track_from_settings(settings)
    master_df = AnalysisModule(input_folder,exp_list).analyze_from_settings(settings)
    return master_df

if __name__ == "__main__":
    from pipeline.settings.settings_dict import settings
    from pathlib import Path
    from os.path import join
    import json
    from tifffile import imwrite, imread

    
    t1 = time()
    input_folder = settings['input_folder']
    master_df = run_pipeline(settings)
    
    t2 = time()
    if t2-t1<60: print(f"Time to process: {round(t2-t1,ndigits=3)} sec\n")
    else: print(f"Time to process: {round((t2-t1)/60,ndigits=1)} min\n")
    # exp_list = PreProcessModule(input_folder,
    #                             **settings['init']).process_from_settings(settings)
    
    # exp_list = SegmentationModule(input_folder,exp_list).segment_from_settings(settings)
    
    # exp_list = TrackingModule(input_folder,exp_list).track_from_settings(settings)
    
    # # Manually created the wound masks
    # parent_fold = '/home/Test_images/szimi/MET/20240515-fMLF_diffusion'
    # img_folds = list(Path(parent_fold).glob('**/*_s1'))
    # mask = imread('/home/Test_images/szimi/MET/Mask.tif')
    # # print(cpu_count())
    # for img_fold in img_folds:
    #     with open(join(str(img_fold),"exp_settings.json"),'r') as file:
    #         exp_set = json.load(file)
    #     frames = exp_set['img_properties']['n_frames']
    #     save_path = join(str(img_fold),"Masks_wound")
    #     Path(save_path).mkdir(exist_ok=True)
    #     for i in range(frames):
    #         mask_name = f"{save_path}/YFP_s01_f{i+1:04d}_z0001.tif"
    #         imwrite(mask_name, mask.astype('uint16'))
    #     exp_set['analysis']["is_reference_masks"] = True
    #     exp_set['analysis']["reference_masks"] = {"wound":{'fold_src':"Images_Registered",'channel_show':"YFP"}}
    #     exp_set['analysis']['um_per_pixel'] = (0.649,0.649)
    #     with open(join(str(img_fold),"exp_settings.json"), 'w') as fp:
    #         json.dump(exp_set, fp, indent=4)
    
    # master_df = AnalysisModule(input_folder,exp_list).analyze_from_settings(settings)