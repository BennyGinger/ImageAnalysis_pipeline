from __future__ import annotations
from os import PathLike
from image_handeling.Experiment_Classes import Experiment
from image_handeling.data_utility import img_list_src, create_save_folder, save_tif
from tifffile import imread
from cv2 import GaussianBlur
import numpy as np
from concurrent.futures import ThreadPoolExecutor


def apply_blur(input_dict: dict)-> None:
    img = imread(input_dict['img_path'])
    # Blur image and save
    blur_img = GaussianBlur(img,**input_dict['blur']).astype(np.uint16)
    save_tif(blur_img,input_dict['save_path'],**input_dict['metadata'])

# # # # # # # # main functions # # # # # # # # # 
def blur_img(exp_set_list: list[Experiment], blur_kernel: list[int], blur_sigma: int, img_fold_src: PathLike="", blur_overwrite: bool = False)-> None:
    # Check if kernel contains 2 odd intengers >= to 3
    if not all(i%2!=0 for i in blur_kernel) and not all(i>=3 for i in blur_kernel):
        print("The input 'blur_kernel' must contain 2 odd intengers greater or equal to 3")

    # Get the exp_path and load exp_para
    for exp_set in exp_set_list:
        # Check if exists
        if exp_set.process.img_blured and not blur_overwrite:
            # Log
            print(f" --> Images are already blured with {exp_set.process.img_blured}")
            continue
        
        # Log
        print(f" --> Bluring images using a kernel of {blur_kernel} and sigma of {blur_sigma}")
        img_list = img_list_src(exp_set, img_fold_src)
        input_data = [{'img_path':path,
                     'save_path':path.replace("Images","Images_Blured").replace('_Registered',''),
                     'blur':{'ksize':blur_kernel,
                             'sigmaX':blur_sigma},
                     'metadata':{'um_per_pixel':exp_set.analysis.um_per_pixel,
                                 'finterval':exp_set.analysis.interval_sec}}
                    for path in img_list]
        
        # Create blur dir
        create_save_folder(exp_set.exp_path,'Images_Blured')
        
        with ThreadPoolExecutor() as executor:
            executor.map(apply_blur,input_data)
            
        # Save settings
        exp_set.process.img_blured = [f"blur_kernel={blur_kernel}",f"blur_sigma={blur_sigma}"]
        exp_set.save_as_json()
    return exp_set_list