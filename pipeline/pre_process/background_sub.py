from __future__ import annotations
from image_handeling.Experiment_Classes import Experiment
from concurrent.futures import ProcessPoolExecutor
from tifffile import imread
from smo import SMO
from image_handeling.data_utility import save_tif


################################## main function ###################################
def background_sub(exp_set_list: list[Experiment], sigma: float=0.0, size: int=7, bg_sub_overwrite: bool=False)-> list[Experiment]:
    """For each experiment, apply a background substraction on the images and return a list of Settings objects"""
    for exp_set in exp_set_list:
        if exp_set.process.background_sub and not bg_sub_overwrite:
            print(f" --> Background substraction was already applied to the images with {exp_set.process.background_sub}")
            continue
        print(f" --> Applying background substraction to the images with sigma={sigma} and size={size}")
        
        # Add smo_object to img_path
        smo = SMO(shape=(exp_set.img_properties.img_width,exp_set.img_properties.img_length),sigma=sigma,size=size)
        input_data = [{'img_path':path,
                       'smo':smo,
                       'metadata':{'um_per_pixel':exp_set.analysis.um_per_pixel,
                                   'finterval':exp_set.analysis.interval_sec}}
                      for path in exp_set.raw_imgs_lst]
        
        with ProcessPoolExecutor() as executor:
            executor.map(apply_bg_sub,input_data)
            
        exp_set.process.background_sub = (f"sigma={sigma}",f"size={size}")
        exp_set.save_as_json()
    return exp_set_list

################################ Satelite functions ################################
def apply_bg_sub(input_dict: dict)-> None:
    # Initiate SMO
    img = imread(input_dict['img_path'])
    bg_img = input_dict['smo'].bg_corrected(img)
    # Reset neg val to 0
    bg_img[bg_img<0] = 0
    save_tif(bg_img,input_dict['img_path'],**input_dict['metadata'])