from __future__ import annotations



from dataclasses import dataclass, field
from importlib.metadata import version
from os import PathLike, sep, walk
from os.path import join
from re import search

from tqdm import tqdm

from pipeline.pre_process.image_sequence import create_img_seq
from pipeline.image_handeling.Experiment_Classes import Experiment, init_from_dict, init_from_json
from pipeline.image_handeling.Base_Module_Class import BaseModule


EXTENTION = ('.nd2','.tif','.tiff')

@dataclass
class ImageExtractionModule(BaseModule):
    # Attributes from the BaseModule class:
        # input_folder: PathLike | list[PathLike]
        # exp_obj_lst: list[Experiment] = field(init=False)
    active_channel_list: list[str] | str = field(default_factory=list)
    full_channel_list: list[str] | str = field(default_factory=list)
    overwrite: bool = False
    
    def __post_init__(self) -> None:
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
        print("\n\033[93mExtracting images =====\033[0m")
        print(f"... Searching for {EXTENTION} files in {self.input_folder} ...")
        # Get the path of all the nd2 files in all subsequent folders/subfolders and exp_dict if available
        if isinstance(self.input_folder,str):
            return self.get_img_path(self.input_folder)
        
        if isinstance(self.input_folder,list):
            exp_files = []
            for folder in self.input_folder:
                exp_files.extend(self.get_img_path(folder))
            return exp_files
    
    def extract_img_seq(self, img_path_list: list[PathLike])-> list[Experiment]:
        # Extract the image sequence from the image files, return metadata dict for each exp (i.e. each serie in the image file)
        metadata_lst: list[PathLike | dict] = []
        for img_path in tqdm(img_path_list,desc="\033[94mExperiments\033[0m",colour='blue'):
            print("\n")
            metadata_lst.extend(create_img_seq(img_path,self.active_channel_list,self.full_channel_list,self.overwrite))
        
        # Initiate the Experiment object
        exp_objs = [self.init_exp_obj(meta) for meta in metadata_lst]
        # Save the settings
        for exp_obj in exp_objs:
            # Add the version of the pipeline
            exp_obj.version = version('ImageAnalysis')
            exp_obj.save_as_json()
        return exp_objs
    
    def get_labels(self, exp_obj: Experiment)-> list[str]:
        # Get the path of upstream of the input folder, i.e. minus the folder name 
        parent_path = self.input_folder.rsplit(sep,1)[0]
        # Remove the parent path from the image path
        exp_path = exp_obj.exp_path.replace(parent_path,'')
        # Return all the folders in the path as labels, except the first (empty) and last one (the file name)
        return exp_path.split(sep)[1:-1]
    
    @staticmethod
    def get_img_path(folder: PathLike)-> list[PathLike]:
        # Get the path of all the nd2 files in all subsequent folders/subfolders and exp_dict if available
        imgS_path = []
        for root , _, files in walk(folder):
            for f in files:
                # Look for all files with selected extension and that are not already processed 
                if not search(r'_f\d\d\d',f) and f.endswith(EXTENTION):
                    imgS_path.append(join(root,f))
        return sorted(imgS_path) 
    
    @staticmethod
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