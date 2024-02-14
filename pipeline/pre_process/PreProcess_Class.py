from __future__ import annotations
from dataclasses import dataclass, field
from os.path import join
from os import sep, walk
from re import search
from image_sequence import img_seq_all
from image_blur import blur_img
from background_sub import background_sub
from image_registration import register_img, channel_shift_register
from settings.Setting_Class import Settings
from Experiment_Classes import Experiment

EXTENTION = ('.nd2','.tif','.tiff')

@dataclass
class PreProcess:
    input_folder: str | list[str]
    active_channel_list: list[str]
    full_channel_list: list[str] = field(default_factory=list)
    overwrite: bool = False
    experiment_list: list[Experiment] = field(default_factory=list)

    def __post_init__(self)-> None:
        # Initialize the experiment list
        print("Initializing the PrePocess Module")
        img_path_list = self.gather_all_images()
        ## Set up the full channel list if not provided
        if not self.full_channel_list:
            self.full_channel_list = self.active_channel_list
        ## Convert the images to img_seq
        self.experiment_list = self.convert_to_img_seq(img_path_list)
        
    def gather_all_images(self)-> list[str]:
        # look through the folder and collect all image files
        print(f"\nSearching for {EXTENTION} files in {self.input_folder}")
        # Get the path of all the nd2 files in all subsequent folders/subfolders and exp_dict if available
        if isinstance(self.input_folder,str):
            return get_img_path(self.input_folder)
        
        if isinstance(self.input_folder,list):
            img_path_list = []
            for folder in self.input_folder:
                img_path_list.extend(get_img_path(folder))
            return img_path_list
    
    def convert_to_img_seq(self, img_path_list: list[str], overwrite: bool=False)-> list[Experiment]:
        return img_seq_all(img_path_list,self.active_channel_list,self.full_channel_list,overwrite)
    
    def process_from_settings(self, settings: dict)-> list[Experiment]:
        sets = Settings(settings)
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
    
    def blur(self, kernel: tuple[int], sigma: int, img_fold_src: str=None, overwrite: bool=False)-> list[Experiment]:
        return blur_img(self.experiment_list,kernel,sigma,img_fold_src,overwrite)


################# Helper functions #################
def get_img_path(folder: str)-> list[str]:
    # Get the path of all the nd2 files in all subsequent folders/subfolders and exp_dict if available
    imgS_path = []
    for root , _, files in walk(folder):
        for f in files:
            # Look for all files with selected extension and that are not already processed 
            if not search(r'_f\d\d\d',f) and f.endswith(EXTENTION):
                imgS_path.append(join(sep,root+sep,f))
    return sorted(imgS_path)


def gather_all_images(input_folder: str | list[str])-> list[str]:
    # look through the folder and collect all image files
    print(f"\nSearching for {EXTENTION} files in {input_folder}")
    # Get the path of all the nd2 files in all subsequent folders/subfolders and exp_dict if available
    if isinstance(input_folder,str):
        return get_img_path(input_folder)
    
    if isinstance(input_folder,list):
        img_path_list = []
        for folder in input_folder:
            img_path_list.extend(get_img_path(folder))
        return img_path_list


# @dataclass
# class BgSub:
#     sigma: float = field(default_factory=float)
#     size: int = field(default_factory=int)
#     overwrite: bool = field(default_factory=bool)
    
# @dataclass
# class ChanShift:
#     reg_channel: str = field(default_factory=str)
#     reg_mtd: str = field(default_factory=str)
#     overwrite: bool = field(default_factory=bool)
    
# @dataclass
# class Register:
#     reg_channel: str = field(default_factory=str)
#     reg_mtd: str = field(default_factory=str)
#     reg_ref: str = field(default_factory=str)
#     overwrite: bool = field(default_factory=bool)
    
# @dataclass
# class Blur:
#     kernel: tuple[int] = field(default_factory=tuple)
#     sigma: int = field(default_factory=int)
#     img_fold_src: str = field(default_factory=str)
#     overwrite: bool = field(default_factory=bool)


# @dataclass
# class Settings:
#     settings: dict
#     bg_sub: dict = field(init=False)
#     chan_shift: dict = field(init=False)
#     register: dict = field(init=False)
#     blur: dict = field(init=False)
    
#     def __post_init__(self)-> None:
#         if self.settings['run_bg_sub']:
#             self.bg_sub = self.settings['bg_sub']
#         if self.settings['run_chan_shift']:
#             self.chan_shift = self.settings['chan_shift']
#         if self.settings['run_register']:
#             self.register = self.settings['register']
#         if self.settings['run_blur']:
#             self.blur = self.settings['blur']
#         self.update_overwrite()
        
#     def update_overwrite(self)-> None:
#         active_branches = [f.name for f in fields(self) if hasattr(self,f.name) and f.name != 'settings']
#         current_overwrite = [getattr(self,f)['overwrite'] for f in active_branches]

#         # Get the new overwrite list, if the previous is true then change the next to true, else keep the same
#         new_overwrite = []; is_False = True
#         for i in range(len(current_overwrite)):
#             if current_overwrite[i] == False and is_False:
#                 new_overwrite.append(current_overwrite[i])
#             elif current_overwrite[i] == True and is_False:
#                 new_overwrite.append(current_overwrite[i])
#                 is_False = False
#             elif not is_False:
#                 new_overwrite.append(True)# Update the overwrite attribute
        
#         # Update the overwrite attribute
#         for i,branch in enumerate(active_branches):
#             temp_dict = getattr(self,branch)
#             temp_dict['overwrite'] = new_overwrite[i]
#             setattr(self,branch,temp_dict)


