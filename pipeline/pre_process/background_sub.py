from __future__ import annotations
from os import PathLike, sep

from concurrent.futures import ThreadPoolExecutor
from tifffile import imread
from smo import SMO
from functools import partial
from pipeline.image_handeling.data_utility import save_tif


################################## main function ###################################
def background_sub(img_paths: list[PathLike], sigma: float=0.0, size: int=7, 
                   um_per_pixel: tuple[float,float]=None, finterval: int=None)-> None:
    """For each experiment, apply a background substraction on the images and return a list of Settings objects"""
    # Log
    exp_name = img_paths[0].rsplit(sep,2)[0].rsplit(sep,1)[1]
    print(f" --> Applying background substraction to {exp_name} with sigma={sigma} and size={size}")
    # Generate input data
    img_shape = imread(img_paths[0]).shape
    smo = SMO(shape=img_shape,sigma=sigma,size=size) 
    apply_bg_sub_partial = partial(apply_bg_sub,smo=smo,
                                   metadata={'um_per_pixel':um_per_pixel,'finterval':finterval})
    
    # Apply background substraction
    with ThreadPoolExecutor() as executor:
        executor.map(apply_bg_sub_partial,img_paths)


################################ Satelite functions ################################
def apply_bg_sub(img_path: PathLike, smo: SMO, metadata: dict)-> None:
    # Read image
    img_array = imread(img_path)
    # Initiate SMO
    bg_img = smo.bg_corrected(img_array)
    # Reset neg val to 0
    bg_img[bg_img<0] = 0
    save_tif(bg_img,img_path,**metadata)


if __name__ == "__main__":
    from os import listdir
    from os.path import join
    from time import time
    # Test
    
    folder = '/home/Test_images/nd2/Run2/c2z25t23v1_nd2_s1/Images'
    img_paths = [join(folder,file) for file in listdir(folder)]
    start = time()
    background_sub(img_paths)
    end = time()
    print(f"Time taken: {end-start}")