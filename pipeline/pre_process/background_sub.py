from __future__ import annotations
from image_handeling.Experiment_Classes import Experiment
from concurrent.futures import ProcessPoolExecutor
from tifffile import imread
from smo import SMO
from image_handeling.data_utility import save_tif, is_processed


################################## main function ###################################
def background_sub(exp_obj_lst: list[Experiment], sigma: float=0.0, size: int=7, overwrite: bool=False)-> list[Experiment]:
    """For each experiment, apply a background substraction on the images and return a list of Settings objects"""
    for exp_obj in exp_obj_lst:
        # Activate the branch
        exp_obj.preprocess.is_background_sub = True
        # Already processed?
        if is_processed(exp_obj.preprocess.background_sub,overwrite=overwrite):
            print(f" --> Background substraction was already applied to the images with {exp_obj.preprocess.background_sub}")
            continue
        # Apply background substraction
        print(f" --> Applying background substraction to the images with sigma={sigma} and size={size}")
        # Generate input data
        smo = SMO(shape=(exp_obj.img_properties.img_width,exp_obj.img_properties.img_length),sigma=sigma,size=size)
        input_data = [{'img_path':path,
                       'smo':smo,
                       'metadata':{'um_per_pixel':exp_obj.analysis.um_per_pixel,
                                   'finterval':exp_obj.analysis.interval_sec}}
                      for path in exp_obj.ori_imgs_lst]
        
        with ProcessPoolExecutor() as executor:
            executor.map(apply_bg_sub,input_data)
            
        exp_obj.preprocess.background_sub = (f"sigma={sigma}",f"size={size}")
        exp_obj.save_as_json()
    return exp_obj_lst

################################ Satelite functions ################################
def apply_bg_sub(input_dict: dict)-> None:
    # Initiate SMO
    img = imread(input_dict['img_path'])
    bg_img = input_dict['smo'].bg_corrected(img)
    # Reset neg val to 0
    bg_img[bg_img<0] = 0
    save_tif(bg_img,input_dict['img_path'],**input_dict['metadata'])