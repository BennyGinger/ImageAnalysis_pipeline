from __future__ import annotations
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from time import sleep
from pipeline.pre_process.image_blur import blur_images
from pipeline.pre_process.background_sub import background_sub
from pipeline.pre_process.image_registration import correct_frame_shift, correct_channel_shift
from pipeline.settings.Setting_Classes import Settings
from pipeline.utilities.Experiment_Classes import Experiment, init_from_json
from pipeline.utilities.Base_Module_Class import BaseModule
from pipeline.utilities.data_utility import img_list_src, is_processed

EXTENTION = ('.nd2','.tif','.tiff')

@dataclass
class PreProcessModule(BaseModule):
    # Attributes from the BaseModule class:
        # input_folder: PathLike | list[PathLike]
        # exp_obj_lst: list[Experiment] = field(init=False)
    def __post_init__(self)-> None:
        super().__post_init__()
        if self.exp_obj_lst:
            return
        
        # Initialize the experiment list
        jsons_path = self.gather_all_json_path()
        self.exp_obj_lst = [init_from_json(json_path) for json_path in jsons_path]
            
    def process_from_settings(self, settings: dict)-> list[Experiment]:
        # Process the images based on the settings
        sets = Settings(settings)
        if not hasattr(sets,'preprocess'):
            print("\n\033[93mNo preprocess settings found =====\033[0m")
            self.save_as_json()
            return self.exp_obj_lst
        sets = sets.preprocess
        # Run the different pre-process functions
        print("\n\033[93mPreprocessing images =====\033[0m")
        if hasattr(sets,'bg_sub'):
            self.bg_sub(**sets.bg_sub)
        if hasattr(sets,'chan_shift'):
            self.channel_shift(**sets.chan_shift)
        if hasattr(sets,'frame_shift'):
            self.frame_shift(**sets.frame_shift)
        if hasattr(sets,'blur'):
            self.blur(**sets.blur)
        print("\n\033[93mPreprocess done =====\033[0m")
        self.save_as_json()
        return self.exp_obj_lst
    
    @staticmethod
    def _bg_sub(exp_obj: Experiment, sigma: float, size: int, overwrite: bool)-> None:
        
        # Activate the branch
        exp_obj.preprocess.is_background_sub = True
        # Already processed?
        if is_processed(exp_obj.preprocess.background_sub,overwrite=overwrite):
            print(f" --> Background substraction was already applied to the images with {exp_obj.preprocess.background_sub}")
            sleep(0.1)
            return
        # Apply background substraction
        background_sub(exp_obj.ori_imgs_lst,sigma,size,exp_obj.analysis.um_per_pixel,exp_obj.analysis.interval_sec)
        # Save the settings
        exp_obj.preprocess.background_sub = (f"sigma={sigma}",f"size={size}","fold_src=Images")
        exp_obj.save_as_json()
    
    def bg_sub(self, sigma: float=0, size: int=7, overwrite: bool=False)-> None:
        """Method to apply background substraction to the images. 
        The images are saved in the same folder as the original images.
        """
        print("\n-> Removing background from images")
        self._loop_over_exp(self._bg_sub,sigma=sigma,size=size,overwrite=overwrite)
    
    @staticmethod
    def _channel_shift(exp_obj: Experiment, reg_channel: str, reg_mtd: str, overwrite: bool)-> None:
        
        exp_name = Path(exp_obj.exp_path).stem
        if len(exp_obj.active_channel_list)==1:
            print(f" --> Only one channel in {exp_name}, no channel shift correction needed")
            return
        
        # Activate the branch
        exp_obj.preprocess.is_channel_reg = True
        
        if is_processed(exp_obj.preprocess.channel_reg,overwrite=overwrite):
            print(f" --> Channel shift correction was already applied to {exp_name} with {exp_obj.preprocess.channel_reg}")
            sleep(0.1)
            return
        
        # Get the images to register
        img_fold_src,img_paths = img_list_src(exp_obj,'Images')
        um_per_pixel = exp_obj.analysis.um_per_pixel
        finterval = exp_obj.analysis.interval_sec
        
        # Apply the channel shift
        correct_channel_shift(img_paths,reg_mtd,reg_channel,exp_obj.active_channel_list,um_per_pixel,finterval)
        
        # Save the settings
        exp_obj.preprocess.channel_reg = [f"reg_mtd={reg_mtd}",f"reg_channel={reg_channel}",f"fold_src={img_fold_src}"]
        exp_obj.save_as_json()
    
    def channel_shift(self, reg_channel: str, reg_mtd: str, overwrite: bool=False)-> None:
        """Method to apply channel shift to the images. Images are saved in the same folder as 
        the original images.
        """
        print("\n-> Correcting channel shift in images")   
        self._loop_over_exp(self._channel_shift,reg_channel=reg_channel,reg_mtd=reg_mtd,overwrite=overwrite)
    
    @staticmethod
    def _frame_shift(exp_obj: Experiment, reg_channel: str, reg_mtd: str, img_ref: str, overwrite: bool)-> None:
        exp_name = Path(exp_obj.exp_path).stem
        if exp_obj.img_properties.n_frames==1:
            print(f" --> Only one frame in {exp_name}, no frame shift correction needed")
            return
        
        # Activate the branch
        exp_obj.preprocess.is_frame_reg = True
        
        # Already processed?
        if is_processed(exp_obj.preprocess.frame_reg,overwrite=overwrite):
            print(f" --> Frame shift correction was already applied to {exp_name} with {exp_obj.preprocess.frame_reg}")
            sleep(0.1)
            return
        
        # Get the images to register
        img_fold_src,img_paths = img_list_src(exp_obj,'Images')
        um_per_pixel = exp_obj.analysis.um_per_pixel
        finterval = exp_obj.analysis.interval_sec
        frames = exp_obj.img_properties.n_frames
        
        # Apply the frame shift
        correct_frame_shift(img_paths,reg_channel,reg_mtd,img_ref,overwrite,um_per_pixel,finterval,frames)
        
        # Save settings
        exp_obj.preprocess.frame_reg = [f"reg_channel={reg_channel}",f"reg_mtd={reg_mtd}",f"img_ref={img_ref}",f"fold_src={img_fold_src}"]
        exp_obj.save_as_json()
            
    def frame_shift(self, reg_channel: str, reg_mtd: str, img_ref: str, overwrite: bool=False)-> None:
        """Method to apply frame shift to the images. Images are saved in a separate folder,
        named 'Images_Registered'.
        """
        print("\n-> Correcting frame shift in images")
        self._loop_over_exp(self._frame_shift,reg_channel=reg_channel,reg_mtd=reg_mtd,img_ref=img_ref,overwrite=overwrite)
    
    @staticmethod
    def _blur(exp_obj: Experiment, sigma: int, img_fold_src: PathLike, kernel: tuple[int,int], overwrite: bool)-> None:
        # Get the images to blur and the metadata
        img_fold_src,img_paths = img_list_src(exp_obj,img_fold_src)
        
        # Activate the branch, after checking if the images are already processed
        exp_obj.preprocess.is_img_blured = True
        
        # Apply the blur
        blur_images(img_paths,sigma,kernel,exp_obj.analysis.um_per_pixel,exp_obj.analysis.interval_sec,overwrite)
        
        # Save settings
        exp_obj.preprocess.img_blured = [f"blur_kernel={kernel}",f"blur_sigma={sigma}",f"fold_src={img_fold_src}"]
        exp_obj.save_as_json()
    
    def blur(self, sigma: int, img_fold_src: PathLike="", kernel: tuple[int,int]=(15,15), overwrite: bool=False)-> None:
        """Method to apply blur to the images. Images are saved in a separate folder"""
        
        print("\n-> Blurring images")
        self._loop_over_exp(self._blur,sigma=sigma,img_fold_src=img_fold_src,kernel=kernel,overwrite=overwrite)
             
    


if __name__ == "__main__":
    

    img_path = '/home/Test_images/nd2/Run3'
    obj = PreProcessModule(img_path,overwrite=True)
    print(obj.exp_obj_lst)
