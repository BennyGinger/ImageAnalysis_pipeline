from __future__ import annotations
from dataclasses import dataclass, field
from os.path import join
from os import sep, walk, PathLike
from re import search
from .image_sequence import img_seq_all
from .image_blur import blur_img
from .background_sub import background_sub
from .image_registration import register_img, channel_shift_register
from settings.Setting_Classes import PreProcessSettings
from image_handeling.Experiment_Classes import Experiment
from image_handeling.Base_Module_Class import BaseModule

EXTENTION = ('.nd2','.tif','.tiff')

@dataclass
class PreProcess(BaseModule):
    # Attributes from the BaseModule class:
        # input_folder: PathLike | list[PathLike]
        # experiment_list: list[Experiment] = field(init=False)
    active_channel_list: list[str] = field(default_factory=list)
    full_channel_list: list[str] = field(default_factory=list)
    overwrite: bool = False

    def __post_init__(self)-> None:
        super().__post_init__()
        img_path_list = self.gather_all_images()
        print('done')
        ## Set up the full channel list if not provided
        if not self.full_channel_list:
            self.full_channel_list = self.active_channel_list
        ## Convert the images to img_seq
        self.experiment_list = self.convert_to_img_seq(img_path_list,self.overwrite)
        
    def gather_all_images(self)-> list[str]:
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
    
    def convert_to_img_seq(self, img_path_list: list[PathLike], overwrite: bool=False)-> list[Experiment]:
        return img_seq_all(img_path_list,self.active_channel_list,self.full_channel_list,overwrite)
    
    def process_from_settings(self, settings: dict)-> list[Experiment]:
        sets = PreProcessSettings(settings)
        if self.overwrite:
            sets.update_overwrite(overwrite_all=True)
        
        if hasattr(sets,'bg_sub'):
            self.experiment_list = self.bg_sub(**sets.bg_sub)
        if hasattr(sets,'chan_shift'):
            self.experiment_list = self.chan_shift(**sets.chan_shift)
        if hasattr(sets,'register'):
            self.experiment_list = self.register(**sets.register)
        if hasattr(sets,'blur'):
            self.experiment_list = self.blur(**sets.blur)
        return self.experiment_list
    
    def bg_sub(self, sigma: float=0, size: int=7, overwrite: bool=False)-> list[Experiment]:
        return background_sub(self.experiment_list,sigma,size,overwrite)
    
    def chan_shift(self, reg_channel: str, reg_mtd: str, overwrite: bool=False)-> list[Experiment]:
        return channel_shift_register(self.experiment_list,reg_mtd,reg_channel,overwrite)
    
    def register(self, reg_channel: str, reg_mtd: str, reg_ref: str, overwrite: bool=False)-> list[Experiment]:
        return register_img(self.experiment_list,reg_channel,reg_mtd,reg_ref,overwrite)
    
    def blur(self, kernel: tuple[int], sigma: int, img_fold_src: PathLike="", overwrite: bool=False)-> list[Experiment]:
        return blur_img(self.experiment_list,kernel,sigma,img_fold_src,overwrite)


def get_img_path(folder: PathLike)-> list[PathLike]:
    # Get the path of all the nd2 files in all subsequent folders/subfolders and exp_dict if available
    imgS_path = []
    for root , _, files in walk(folder):
        for f in files:
            # Look for all files with selected extension and that are not already processed 
            if not search(r'_f\d\d\d',f) and f.endswith(EXTENTION):
                imgS_path.append(join(sep,root+sep,f))
    return sorted(imgS_path) 



