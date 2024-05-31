from __future__ import annotations
from os import PathLike, scandir, sep
from pathlib import Path
from time import sleep
from pipeline.image_handeling.data_utility import create_save_folder, save_tif, run_multithread
from tifffile import imread
from cv2 import GaussianBlur
import numpy as np


################################## main function ###################################
def blur_images(img_paths: list[PathLike], sigma: int, kernel: tuple[int,int]=(15,15), um_per_pixel: tuple[float,float]=None, finterval: int=None, overwrite: bool=False)-> None:
    # Check if kernel contains 2 odd intengers >= to 3
    if not all(i%2!=0 for i in kernel) and not all(i>=3 for i in kernel):
        raise ValueError("The input 'blur_kernel' must contain 2 odd intengers greater or equal to 3")
    
    # Already processed
    exp_path = str(Path(img_paths[0]).parent).rsplit(sep,1)[0]
    print(f" --> Bluring images in \033[94m{exp_path}\033[0m")
    save_path = create_save_folder(exp_path,'Images_Blured')
    if any(scandir(save_path)) and not overwrite:
        print(f"  ---> Images are already blured with kernel={kernel} and sigma={sigma}")
        sleep(0.1)
        return

    # Get the metadata and fixed parameters
    print(f"  ---> Bluring images using a kernel of {kernel} and sigma of {sigma}")
    fixed_args = {'sigma':sigma,
                  'kernel':kernel,
                  'metadata':{'um_per_pixel':um_per_pixel,
                              'finterval':finterval}}
    
    # Run the bluring process
    run_multithread(apply_blur,img_paths,fixed_args)

################################ Satelite functions ################################
def apply_blur(img_path: Path, sigma: int, kernel: tuple[int,int], metadata: dict)-> None:
    img = imread(img_path)
    # Blur image and save
    blur_img = GaussianBlur(img,kernel,sigma).astype(np.uint16)
    # Save
    save_path = str(img_path).replace("Images","Images_Blured").replace('_Registered','')
    save_tif(blur_img,save_path,**metadata)

if __name__ == "__main__":
    # Test
    from time import time
    
    t1 = time()
    img_paths = list(Path("/home/Test_images/nd2/Run2/c2z25t23v1_nd2_s1/Images_Registered").glob("*.tif"))
    blur_images(img_paths,5,overwrite=True)
    t2 = time()
    print(f"Time to process: {round(t2-t1,ndigits=3)} sec\n")