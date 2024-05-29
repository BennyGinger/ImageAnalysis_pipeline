from __future__ import annotations
from dataclasses import dataclass, field
from os.path import join
from os import sep, walk, PathLike
from re import search
from importlib.metadata import version
from pathlib import Path

from pipeline.pre_process.image_sequence import create_img_seq
from pipeline.pre_process.image_blur import blur_images
from pipeline.pre_process.background_sub import background_sub
from pipeline.pre_process.image_registration import correct_frame_shift, correct_channel_shift
from pipeline.settings.Setting_Classes import Settings
from pipeline.image_handeling.Experiment_Classes import Experiment, init_from_dict, init_from_json
from pipeline.image_handeling.Base_Module_Class import BaseModule
from pipeline.image_handeling.data_utility import img_list_src, is_processed

EXTENTION = ('.nd2','.tif','.tiff')

@dataclass
class PreProcessModule(BaseModule):
    # Attributes from the BaseModule class:
        # input_folder: PathLike | list[PathLike]
        # exp_obj_lst: list[Experiment] = field(init=False)
    active_channel_list: list[str] | str = field(default_factory=list)
    full_channel_list: list[str] | str = field(default_factory=list)
    overwrite: bool = False

    def __post_init__(self)-> None:
        super().__post_init__()
        exp_files = self.search_exp_files()
        # Check if the channel lists are strings, if so convert them to lists
        if isinstance(self.active_channel_list,str):
            self.active_channel_list = [self.active_channel_list]
        if isinstance(self.full_channel_list,str):
            self.full_channel_list = [self.full_channel_list]
        
        ## Convert the images to img_seq
        self.exp_obj_lst = self.extract_img_seq(exp_files)
        
        # Update the tags
        for exp_obj in self.exp_obj_lst:
            exp_obj.analysis.labels = self.get_labels(exp_obj)
            
    def search_exp_files(self)-> list[PathLike]:
        # look through the folder and collect all image files
        print("\nExtracting images =====")
        print(f"... Searching for {EXTENTION} files in {self.input_folder} ...")
        # Get the path of all the nd2 files in all subsequent folders/subfolders and exp_dict if available
        if isinstance(self.input_folder,str):
            return get_img_path(self.input_folder)
        
        if isinstance(self.input_folder,list):
            exp_files = []
            for folder in self.input_folder:
                exp_files.extend(get_img_path(folder))
            return exp_files
    
    def extract_img_seq(self, img_path_list: list[PathLike])-> list[Experiment]:
        # Extract the image sequence from the image files, return metadata dict for each exp (i.e. each serie in the image file)
        metadata_lst: list[PathLike | dict] = []
        for img_path in img_path_list:
            metadata_lst.extend(create_img_seq(img_path,self.active_channel_list,self.full_channel_list,self.overwrite))
        
        # Initiate the Experiment object
        exp_objs = [init_exp_obj(meta) for meta in metadata_lst]
        # Save the settings
        for exp_obj in exp_objs:
            # Add the version of the pipeline
            exp_obj.version = version('ImageAnalysis')
            exp_obj.save_as_json()
        return exp_objs
    
    def process_from_settings(self, settings: dict)-> list[Experiment]:
        # Process the images based on the settings
        sets = Settings(settings)
        if not hasattr(sets,'preprocess'):
            print("\nNo preprocess settings found =====")
            self.save_as_json()
            return self.exp_obj_lst
        sets = sets.preprocess
        # Run the different pre-process functions
        print("\nPreprocessing images =====")
        if hasattr(sets,'bg_sub'):
            self.bg_sub(**sets.bg_sub)
        if hasattr(sets,'chan_shift'):
            self.channel_shift(**sets.chan_shift)
        if hasattr(sets,'frame_shift'):
            self.frame_shift(**sets.frame_shift)
        if hasattr(sets,'blur'):
            self.blur(**sets.blur)
        print("\nPreprocess done =====")
        self.save_as_json()
        return self.exp_obj_lst
    
    def bg_sub(self, sigma: float=0, size: int=7, overwrite: bool=False)-> None:
        """Method to apply background substraction to the images. 
        The images are saved in the same folder as the original images.
        """
        print("\n-> Removing background from images")
        for exp_obj in self.exp_obj_lst:
            # Activate the branch
            exp_obj.preprocess.is_background_sub = True
            # Already processed?
            if is_processed(exp_obj.preprocess.background_sub,overwrite=overwrite):
                print(f" --> Background substraction was already applied to the images with {exp_obj.preprocess.background_sub}")
                continue
            # Apply background substraction
            background_sub(exp_obj.ori_imgs_lst,sigma,size,overwrite)
            # Save the settings
            exp_obj.preprocess.background_sub = (f"sigma={sigma}",f"size={size}","fold_src=Images")
            exp_obj.save_as_json()
    
    def channel_shift(self, reg_channel: str, reg_mtd: str, overwrite: bool=False)-> None:
        """Method to apply channel shift to the images. Images are saved in the same folder as 
        the original images.
        """
        print("\n-> Correcting channel shift in images")   
        for exp_obj in self.exp_obj_lst:
            exp_name = Path(exp_obj.exp_path).stem
            if len(exp_obj.active_channel_list)==1:
                print(f" --> Only one channel in {exp_name}, no channel shift correction needed")
                continue
            
            # Activate the branch
            exp_obj.preprocess.is_channel_reg = True
            
            if is_processed(exp_obj.preprocess.channel_reg,overwrite=overwrite):
                print(f" --> Channel shift correction was already applied to {exp_name} with {exp_obj.preprocess.channel_reg}")
                continue
            
            # Get the images to register
            img_fold_src,img_paths = img_list_src(exp_obj,'Images')
            
            # Apply the channel shift
            correct_channel_shift(img_paths,reg_mtd,reg_channel,exp_obj.active_channel_list,
                                  {'finterval':exp_obj.analysis.interval_sec,'um_per_pixel':exp_obj.analysis.um_per_pixel})
            
            # Save the settings
            exp_obj.preprocess.channel_reg = [f"reg_mtd={reg_mtd}",f"reg_channel={reg_channel}",f"fold_src={img_fold_src}"]
            exp_obj.save_as_json()
    
    def frame_shift(self, reg_channel: str, reg_mtd: str, img_ref: str, overwrite: bool=False)-> None:
        """Method to apply frame shift to the images. Images are saved in a separate folder,
        named 'Images_Registered'.
        """
        print("\n-> Correcting frame shift in images")
        for exp_obj in self.exp_obj_lst:
            exp_name = Path(exp_obj.exp_path).stem
            if exp_obj.img_properties.n_frames==1:
                print(f" --> Only one frame in {exp_name}, no frame shift correction needed")
                continue
            
            # Activate the branch
            exp_obj.preprocess.is_frame_reg = True
            
            # Already processed?
            if is_processed(exp_obj.preprocess.frame_reg,overwrite=overwrite):
                print(f" --> Frame shift correction was already applied to {exp_name} with {exp_obj.preprocess.frame_reg}")
                continue
            
            # Get the images to register
            img_fold_src,img_paths = img_list_src(exp_obj,'Images')
            
            # Apply the frame shift
            correct_frame_shift(img_paths,reg_channel,reg_mtd,img_ref,overwrite,{'finterval':exp_obj.analysis.interval_sec,'um_per_pixel':exp_obj.analysis.um_per_pixel},exp_obj.img_properties.n_frames)
            
            # Save settings
            exp_obj.preprocess.frame_reg = [f"reg_channel={reg_channel}",f"reg_mtd={reg_mtd}",f"img_ref={img_ref}",f"fold_src={img_fold_src}"]
            exp_obj.save_as_json()
            pass
        
        return
    
    def blur(self, sigma: int, img_fold_src: PathLike="", kernel: tuple[int,int]=(15,15), overwrite: bool=False)-> None:
        print("\n-> Blurring images")
        for exp_obj in self.exp_obj_lst:
            # Get the images to blur and the metadata
            img_fold_src,img_paths = img_list_src(exp_obj,img_fold_src)
            metadata = {'um_per_pixel':exp_obj.analysis.um_per_pixel,'finterval':exp_obj.analysis.interval_sec}
            
            # Activate the branch, after checking if the images are already processed
            exp_obj.preprocess.is_img_blured = True
            
            # Apply the blur
            blur_images(img_paths,sigma,kernel,metadata,overwrite)
            
            # Save settings
            exp_obj.preprocess.img_blured = [f"blur_kernel={kernel}",f"blur_sigma={sigma}",f"fold_src={img_fold_src}"]
            exp_obj.save_as_json() 

    def get_labels(self, exp_obj: Experiment)-> list[str]:
        # Get the path of upstream of the input folder, i.e. minus the folder name 
        parent_path = self.input_folder.rsplit(sep,1)[0]
        # Remove the parent path from the image path
        exp_path = exp_obj.exp_path.replace(parent_path,'')
        # Return all the folders in the path as labels, except the first (empty) and last one (the file name)
        return exp_path.split(sep)[1:-1]

def get_img_path(folder: PathLike)-> list[PathLike]:
    # Get the path of all the nd2 files in all subsequent folders/subfolders and exp_dict if available
    imgS_path = []
    for root , _, files in walk(folder):
        for f in files:
            # Look for all files with selected extension and that are not already processed 
            if not search(r'_f\d\d\d',f) and f.endswith(EXTENTION):
                imgS_path.append(join(root,f))
    return sorted(imgS_path) 

def init_exp_obj(metadata: PathLike | str | dict)-> Experiment:
    """Initialize Experiment object from json file if exists, else from the metadata dict. 
    Return the Experiment object."""
    
    if isinstance(metadata,str):
        exp_obj = init_from_json(metadata)
    elif isinstance(metadata,dict):
        exp_obj = init_from_dict(metadata)
        
    # Set all the branch to inactive
    exp_obj.init_to_inactive()
    return exp_obj


if __name__ == "__main__":
    

    img_path = '/home/Test_images/nd2/Run3'
    obj = PreProcessModule(img_path,overwrite=True)
    print(obj.exp_obj_lst)
