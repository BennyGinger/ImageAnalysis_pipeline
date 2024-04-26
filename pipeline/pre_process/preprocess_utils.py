
from __future__ import annotations
from os import PathLike, mkdir
from os.path import join, isdir
from tifffile import imwrite
import numpy as np

def get_resolution(um_per_pixel: tuple[float,float])-> tuple[float,float]:
    x_umpixel,y_umpixel = um_per_pixel
    return 1/x_umpixel,1/y_umpixel

def save_tif(array: np.ndarray, save_path: PathLike, um_per_pixel: tuple[float,float], finterval: int)-> None:
    """Save array as tif with metadata"""
    imagej_metadata = {'finterval':finterval, 'unit': 'um'}
    imwrite(save_path,array.astype(np.uint16),imagej=True,metadata=imagej_metadata,resolution=get_resolution(um_per_pixel))

def create_save_folder(exp_path: PathLike, folder_name: str)-> PathLike:
    save_folder = join(exp_path,folder_name)
    if not isdir(save_folder):
        print(f" ---> Creating saving folder: {save_folder}")
        mkdir(save_folder)
        return save_folder
    print(f" ---> Saving folder already exists: {save_folder}")
    return save_folder