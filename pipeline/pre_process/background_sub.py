from __future__ import annotations
from os import PathLike
from pathlib import Path
from tifffile import imread
from smo import SMO
from pipeline.image_handeling.data_utility import save_tif, run_multithread


################################## main function ###################################
def background_sub(img_paths: list[PathLike], sigma: float=0.0, size: int=7, 
                   um_per_pixel: tuple[float,float]=None, finterval: int=None)-> None:
    """For each experiment, apply a background substraction on the images and return a list of Settings objects"""
    # Log
    exp_path = Path(img_paths[0]).parent.parent
    print(f"--> Applying background substraction to \033[94m{exp_path}\033[0m")
    # Generate input data
    img_shape = imread(img_paths[0]).shape
    fixed_args = {'smo':SMO(shape=img_shape,sigma=sigma,size=size),
                  'metadata':{'um_per_pixel':um_per_pixel,
                              'finterval':finterval}}
    
    # Apply background substraction
    run_multithread(apply_bg_sub,img_paths,fixed_args)


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
    
    folder = '/home/Test_images/szimi/Cellsorter/control/control_001_s1/Images'
    
    img_paths = list(Path(folder).glob('*.tif'))
    
    start = time()
    background_sub(img_paths)
    end = time()
    print(f"Time taken: {end-start}")