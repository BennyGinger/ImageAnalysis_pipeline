from __future__ import annotations
from dataclasses import dataclass, field
from os.path import join, exists
from os import sep, walk, PathLike, scandir
from re import search

from pre_process.metadata import get_metadata
from .image_sequence import process_img
from .image_blur import blur_img
from .background_sub import background_sub
from .image_registration import correct_frame_shift, correct_channel_shift
from settings.Setting_Classes import Settings
from image_handeling.data_utility import create_save_folder
from image_handeling.Experiment_Classes import Experiment, init_from_dict, init_from_json
from image_handeling.Base_Module_Class import BaseModule

EXTENTION = ('.nd2','.tif','.tiff')

@dataclass
class PreProcess(BaseModule):
    # Attributes from the BaseModule class:
        # input_folder: PathLike | list[PathLike]
        # exp_obj_lst: list[Experiment] = field(init=False)
    active_channel_list: list[str] = field(default_factory=list)
    full_channel_list: list[str] = field(default_factory=list)
    overwrite: bool = False

    def __post_init__(self)-> None:
        super().__post_init__()
        exp_files_lst = self.search_exp_files()
        print('done')
        ## Convert the images to img_seq
        self.exp_obj_lst = self.extract_img_seq(exp_files_lst)
        
    def search_exp_files(self)-> list[str]:
        # look through the folder and collect all image files
        print(f"\n... Searching for {EXTENTION} files in {self.input_folder} ...")
        # Get the path of all the nd2 files in all subsequent folders/subfolders and exp_dict if available
        if isinstance(self.input_folder,str):
            return get_img_path(self.input_folder)
        
        if isinstance(self.input_folder,list):
            img_path_list = []
            for folder in self.input_folder:
                img_path_list.extend(get_img_path(folder))
            return img_path_list
    
    def extract_img_seq(self, img_path_list: list[PathLike])-> list[Experiment]:
        exp_list = []
        for img_path in img_path_list:
            exp_list.extend(process_img_file(img_path,self.active_channel_list,self.full_channel_list,self.overwrite))
        return exp_list
    
    def process_from_settings(self, settings: dict)-> list[Experiment]:
        # Process the images based on the settings
        sets = Settings(settings)
        if not hasattr(sets,'preprocess'):
            print("No preprocess settings found")
            self.save_as_json()
            return self.exp_obj_lst
        # Update the settings
        sets = sets.preprocess
        if self.overwrite:
            sets.update_overwrite(overwrite_all=True)
        # Run the different pre-process functions
        if hasattr(sets,'bg_sub'):
            self.exp_obj_lst = self.bg_sub(**sets.bg_sub)
        if hasattr(sets,'chan_shift'):
            self.exp_obj_lst = self.channel_shift(**sets.chan_shift)
        if hasattr(sets,'frame_shift'):
            self.exp_obj_lst = self.frame_shift(**sets.frame_shift)
        if hasattr(sets,'blur'):
            self.exp_obj_lst = self.blur(**sets.blur)
        self.save_as_json()
        return self.exp_obj_lst
    
    def bg_sub(self, sigma: float=0, size: int=7, overwrite: bool=False)-> list[Experiment]:
        return background_sub(self.exp_obj_lst,sigma,size,overwrite)
    
    def channel_shift(self, reg_channel: str, reg_mtd: str, overwrite: bool=False)-> list[Experiment]:
        return correct_channel_shift(self.exp_obj_lst,reg_mtd,reg_channel,overwrite)
    
    def frame_shift(self, reg_channel: str, reg_mtd: str, img_ref: str, overwrite: bool=False)-> list[Experiment]:
        return correct_frame_shift(self.exp_obj_lst,reg_channel,reg_mtd,img_ref,overwrite)
    
    def blur(self, kernel: tuple[int], sigma: int, img_fold_src: PathLike="", overwrite: bool=False)-> list[Experiment]:
        return blur_img(self.exp_obj_lst,kernel,sigma,img_fold_src,overwrite)


def get_img_path(folder: PathLike)-> list[PathLike]:
    # Get the path of all the nd2 files in all subsequent folders/subfolders and exp_dict if available
    imgS_path = []
    for root , _, files in walk(folder):
        for f in files:
            # Look for all files with selected extension and that are not already processed 
            if not search(r'_f\d\d\d',f) and f.endswith(EXTENTION):
                imgS_path.append(join(sep,root+sep,f))
    return sorted(imgS_path) 

def process_img_file(img_path: PathLike, active_channel_list: list[str], 
                     full_channel_list: list[str]=[], overwrite: bool=False)-> list[Experiment]:
    """Extract metadata from img file and process the images to create an image sequence. If multiple series, then each serie is 
    processed individually and an Experiment object is created for each serie. 
    
    Args:
        img_path (PathLike): Path to the image file
        active_channel_list (list[str]): List of active channel names, namly the channels tha need to be processed.
        full_channel_list (list[str], optional): List of all channel names. Defaults to [].
        overwrite (bool, optional): Overwrite the images if they have already been processed. Defaults to False.
        
    Returns:
        list[Experiment]: List of Experiment objects"""
    # Get metadata
    meta_dict = get_metadata(img_path,active_channel_list,full_channel_list)
    
    exp_obj_list = []
    for serie in range(meta_dict['n_series']):
        exp_path = meta_dict['exp_path_list'][serie]
        meta_dict['exp_path'] = exp_path
        print(f"--> Checking exp {exp_path} for image sequence")
        
        # If exp has been processed but removed
        if exists(join(sep,exp_path+sep,'REMOVED_EXP.txt')):
            print(" ---> Exp. has been removed")
            continue
        
        save_folder = create_save_folder(exp_path,'Images')
        
        # If img are already processed
        if any(scandir(save_folder)) and not overwrite:
            print(f" ---> Images have already been converted to image sequence")
            exp_obj = init_exp_obj(exp_path)
            exp_obj_list.append(exp_obj)
            # No need to save the settings as they are already saved
            continue
        
        # If images are not processed, extract imseq and initialize exp_set object
        print(f" ---> Extracting images and converting to image sequence")
        process_img(meta_dict)
        
        exp_obj = init_exp_obj(exp_path,meta_dict)
        exp_obj.save_as_json()
        exp_obj_list.append(exp_obj)
    return exp_obj_list

def init_exp_obj(exp_path: PathLike, meta_dict: dict = None)-> Experiment:
    """Initialize Experiment object from json file if exists, else from the metadata dict. 
    Return the Experiment object."""
    
    if meta_dict:
        meta_dict['exp_path'] = exp_path
        exp_obj = init_from_dict(meta_dict)
    
    if exists(join(sep,exp_path+sep,'exp_settings.json')):
        exp_obj = init_from_json(join(sep,exp_path+sep,'exp_settings.json'))
    else:
        raise FileNotFoundError("No json file found")
    # Set all the branch to inactive
    exp_obj.init_to_inactive()
    return exp_obj



