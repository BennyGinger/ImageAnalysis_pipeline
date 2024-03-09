from __future__ import annotations
from image_handeling.Experiment_Classes import Experiment
from concurrent.futures import ProcessPoolExecutor
from tifffile import imread, imwrite
from smo import SMO
import numpy as np

def apply_bg_sub(processed_image: list)-> None:
    # Initiate SMO
    proc_img_path,smo = processed_image
    img = imread(proc_img_path)
    bg_img = smo.bg_corrected(img)
    # Reset neg val to 0
    bg_img[bg_img<0] = 0
    imwrite(proc_img_path,bg_img.astype(np.uint16))

# # # # # # # # main function # # # # # # # #
def background_sub(exp_set_list: list[Experiment], sigma: float=0.0, size: int=7, bg_sub_overwrite: bool=False)-> list[Experiment]:
    """For each experiment, apply a background substraction on the images and return a list of Settings objects"""
    for exp_set in exp_set_list:
        if exp_set.process.background_sub and not bg_sub_overwrite:
            print(f" --> Background substraction was already applied to the images with {exp_set.process.background_sub}")
            continue
        print(f" --> Applying background substraction to the images with sigma={sigma} and size={size}")
        
        # Add smo_object to img_path
        processed_images_list = exp_set.raw_imgs_lst.copy()
        smo = SMO(shape=(exp_set.img_properties.img_width,exp_set.img_properties.img_length),sigma=sigma,size=size)
        processed_images_list = [(img_path,smo) for img_path in processed_images_list]
        
        with ProcessPoolExecutor() as executor:
            executor.map(apply_bg_sub,processed_images_list)
            
        exp_set.process.background_sub = (f"sigma={sigma}",f"size={size}")
        exp_set.save_as_json()
    return exp_set_list

