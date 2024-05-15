from __future__ import annotations
from functools import partial
import json
import re
from typing import Callable
import numpy as np
from cellpose import models, core
from cellpose.io import logger_setup, masks_flows_to_seg
from os import PathLike, sep
from os.path import isfile, join
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pipeline.image_handeling.data_utility import load_stack, create_save_folder, save_tif

#TODO: replace imwrite with save_tif
MODEL_SETTINGS = {'gpu':core.use_gpu(),
                  'model_type': 'cyto3',
                  'pretrained_model':False,
                  'device':None,
                  'diam_mean':30.,
                  'nchan':2}

CELLPOSE_EVAL = {'batch_size':8,
                 'resample':True,
                 'channels':[0,0],
                 'channel_axis':None,
                 'z_axis':None,
                 'normalize':True,
                 'invert':False,
                 'rescale':None,
                 'diameter':60.,
                 'flow_threshold':0.4,
                 'cellprob_threshold':0.,
                 'do_3D':False,
                 'anisotropy':None,
                 'stitch_threshold':0.,
                 'min_size':15,
                 'niter':None,
                 'augment':False,
                 'tile':True,
                 'tile_overlap':0.1,
                 'bsize':224,
                 'interp':True,
                 'compute_masks':True,
                 'progress':None}

BUILD_IN_MODELS = ['cyto3', 'nuclei', 'cyto2_cp3', 
                'tissuenet_cp3', 'livecell_cp3',
                'yeast_PhC_cp3', 'yeast_BF_cp3',
                'bact_phase_cp3', 'bact_fluor_cp3',
                'deepbacs_cp3', 'cyto2']

IN_HOUSE_MODELS = []


# # # # # # # # main functions # # # # # # # # # 
def cellpose_segmentation(img_paths: list[PathLike], channel_seg: str, model_type: str | PathLike ='cyto3',
                          diameter: float=60., flow_threshold: float=0.4, cellprob_threshold: float=0.0,
                          overwrite: bool=False, process_as_2D: bool=False,
                          save_as_npy: bool=False, **kwargs)-> tuple[dict,dict]:
    """Function to run cellpose segmentation. See https://github.com/MouseLand/cellpose for more details.
    Takes a list of image paths, with names that match the pattern [C]_[S/d{4}]_[F/d{4}]_[Z/d{4}].
    With C the channel name, S the serie number, F the frame number and Z the z_slice number, followed by
    the indicated number of digits. The function will segment the cells in the channel indicated by 'channel_seg'.
    The model type can be a build-in model, a path to a pretrained model or a model from the in-house models.
    A folder named 'Masks_Cellpose' will be created at the same level as the image folder, where the segmented
    masks will be saved. The function will return the model settings and the cellpose evaluation settings.
    
    Args:
        img_paths: list[PathLike] = list of image paths
        channel_seg: str = channel name to segment
        model_type: str | PathLike = model type to use
        diameter: float = Average cell diameter
        flow_threshold: float = Flow threshold for running cellpose
        cellprob_threshold: float = Cell probability threshold for running cellpose
        overwrite: bool = overwrite existing segmentation
        process_as_2D: bool = process as 2D, i.e. perform a maxip on images if they contain z_slices
        save_as_npy: bool = save as npy instead of tif (cellpose function)
        kwargs: dict = additional arguments to pass to the cellpose settings
    
    Returns:
        model_settings: dict = model settings
        cellpose_eval: dict = cellpose evaluation settings"""

    # Set up the segmentation saving folder
    file_type = '.npy' if save_as_npy else '.tif'
    exp_path: PathLike = img_paths[0].rsplit(sep,2)[0]
    save_path = create_save_folder(exp_path,'Masks_Cellpose')
    
    # Check if exist
    if any(channel_seg in file for file in sorted(listdir(save_path)) if file.endswith(file_type)) and not overwrite:
        # Log
        print(f" --> Cells have already been segmented with cellpose as {file_type} for the '{channel_seg}' channel.")
        model,model_settings = load_metadata(exp_path,channel_seg)
        return model,model_settings
    
    # Else run cellpose
    print(f" --> Segmenting cells as {file_type} for the '{channel_seg}' channel")
    
    # Get image property
    frames, z_slices = get_img_prop(img_paths)
    
    # Setup model and eval settings
    nuclear_marker,channels,metadata = unpack_kwargs(kwargs,channel_seg)
    cellpose_setup = CellposeSetup(z_slices,channel_seg,nuclear_marker,process_as_2D,save_as_npy)
    cellpose_eval = cellpose_setup.eval_settings(diameter,flow_threshold,cellprob_threshold,**kwargs)
    model,model_settings = cellpose_setup.setup_model(model_type)
    
    # Run Cellpose
    run_cellpose_partial = partial(run_cellpose,metadata=metadata,img_paths=img_paths,channels=channels,
                                   process_as_2D=process_as_2D,model=model,cellpose_eval=cellpose_eval,
                                   save_as_npy=save_as_npy)
    parallel_executor(run_cellpose_partial,frames,model_settings['gpu'],cellpose_eval['z_axis'])
    return model_settings,cellpose_eval
    

#################### Helper functions ####################
def run_cellpose(frame: int, img_paths: list[PathLike], channels: list[str], process_as_2D: bool,
                 model: models.CellposeModel, cellpose_eval: dict, save_as_npy: bool, metadata: dict)-> None:
    """Function that runs the cellpose segmentation on a single frame of a stack."""
    # Load image/stack and model
    img = load_stack(img_paths,channels,frame,process_as_2D)
    # Save path
    path_name = [path for path in sorted(img_paths) if f'_f{frame+1:04d}' in path][0]
    mask_path = path_name.replace('Images','Masks_Cellpose').replace('_Registered','').replace('_Blured','')
    # log
    print(f"  ---> Processing frame {frame}")
    # Run Cellpose
    masks, flows, _ = model.eval(img,**cellpose_eval)
    # Save
    save_mask(img,masks,flows,model.diam_mean,mask_path,save_as_npy,metadata)

def save_mask(img: np.ndarray | list[np.ndarray], mask: np.ndarray | list[np.ndarray], flows: list[np.ndarray] | list[list],
              diameter: float, mask_path: PathLike, as_npy: bool, metadata: dict)-> None:
    if as_npy:
        if img.ndim==3:
            mask_path = mask_path.replace("_z0001","_allz")
        masks_flows_to_seg(img,mask,flows,mask_path,diameter)
        return
    
    if mask.ndim==3:
        for z_silce in range(mask.shape[0]):
            split_name = mask_path.split("_z")
            mask_path = f"{split_name[0]}_z{z_silce+1:04d}.tif"
            save_tif(mask[z_silce,...].astype('uint16'),mask_path,**metadata)
        return
    save_tif(mask.astype('uint16'),mask_path,**metadata)
    
class CellposeSetup:
    """Class that handles the setup of the cellpose model and eval settings.
    
    Attributes:
        z_slices: int
        channel_seg: str
        nuclear_marker: str = ""
        process_as_2D: bool = False
        save_as_npy: bool = False
        model_settings: dict = MODEL_SETTINGS.copy()
        cellpose_eval: dict = CELLPOSE_EVAL.copy()
        model: models.CellposeModel
    Returns:
        model_settings: dict
        cellpose_eval: dict
    """
    def __init__(self, z_slices: int, channel_seg: str, nuclear_marker: str="", 
                 process_as_2D: bool=False, save_as_npy: bool=False) -> None:
        self.z_slices = z_slices
        self.model_settings: dict = MODEL_SETTINGS.copy()
        self.cellpose_eval: dict = CELLPOSE_EVAL.copy()
        self.channel_seg = channel_seg
        self.nuclear_marker = nuclear_marker
        self.process_as_2D = process_as_2D
        self.save_as_npy = save_as_npy
    
    def eval_settings(self, diameter: float=60., flow_threshold: float=0.4, cellprob_threshold: float=0.0, **kwargs)-> dict:
        # Update the default settings
        self.cellpose_eval['diameter'] = diameter
        self.cellpose_eval['flow_threshold'] = flow_threshold
        self.cellpose_eval['cellprob_threshold'] = cellprob_threshold
        
        # Unpack kwargs
        if kwargs:
            for k,v in kwargs.items():
                self.cellpose_eval[k] = v
        
        if self.nuclear_marker:
            self.cellpose_eval['channels'] = [1,2]
    
        # If 2D or 3D to be treated as 2D
        if self.z_slices==1 or self.process_as_2D:
            return self.cellpose_eval
        
        # If 3D
        self.cellpose_eval['z_axis'] = 0
        if self.cellpose_eval['stitch_threshold']==0:
            self.cellpose_eval['do_3D'] = True
            self.cellpose_eval['anisotropy'] = 2.0
        return self.cellpose_eval
    
    def setup_model(self, model_type: str | PathLike ='cyto3', **kwargs)-> tuple[models.CellposeModel,dict]:
        # Setup log
        logger_setup()
        # Unpack kwargs
        if kwargs:
            for k,v in kwargs.items():
                self.model_settings[k] = v
        # Update cellpose model
        if model_type in BUILD_IN_MODELS:
            self.model_settings['model_type'] = model_type
            self.model = models.CellposeModel(**self.model_settings)
            return self.model,self.model_settings
        if model_type in IN_HOUSE_MODELS:
            self.model_settings['pretrained_model'] = model_type
            self.model_settings['model_type'] = False
            self.model = models.CellposeModel(**self.model_settings)
            return self.model,self.model_settings
        if isfile(model_type):
            self.model_settings['pretrained_model'] = model_type
            self.model_settings['model_type'] = None
            self.model = models.CellposeModel(**self.model_settings)
            return self.model,self.model_settings
        # Else raise error
        raise ValueError(f"Model type '{model_type}' not recognized.\
                          Please choose one of the following: {BUILD_IN_MODELS}\
                          or provide a path to a pretrained model.")

def parallel_executor(func: Callable, frames: int, gpu: bool, z_axis: int)-> None:
    if gpu and z_axis==None: # If GPU and 2D images: parallelization
        with ThreadPoolExecutor() as executor:
            executor.map(func,range(frames))
        return
    
    if gpu and z_axis==0: # If GPU and 3D images: no parallelization
        for frame in range(frames):
            func(frame)
        return
    # If CPU 2D or 3D: parallelization
    with ProcessPoolExecutor() as executor:
        executor.map(func,range(frames))

def get_img_prop(img_paths: list[PathLike])-> tuple[int,int]:
    """Function that returns the number of frames and z_slices of a folder containing images.
    The names of those images must match the pattern [C]_[S/d{1}]_[F/d{4}]_[Z/d{4}],
    where C is a label of the channel name (str), S the serie number followed by 2 digits,
    F the frames followed by 4 digits and Z the z_slices followed by 4 digits"""
    # Gather all frames and all z_slice
    frames = [re.search('_f\d{4}', path).group() 
                     for path in img_paths if re.search('_f\d{4}', path)]
    z_slices = [re.search('_z\d{4}', path).group() 
                     for path in img_paths if re.search('_z\d{4}', path)]
    return len(set(frames)),len(set(z_slices))

def load_metadata(file_path: PathLike, channel_to_seg: str)-> tuple[dict,dict]:
    """Function to load the metadata from the json file if it exists. Experiment obj are saved as json files,
    which contains the metadata for the experiment. The metadata contains the model settings and the cellpose 
    evaluation settings."""
    
    # Check if metadata exists, if not return empty settings
    file_path = join(file_path,"exp_settings.json")
    if not isfile(file_path):
        print(f" ---> No metadata found for the '{channel_to_seg}' channel.")
        return {},{}
    # Load metadata
    print(f" ---> Loading metadata for the '{channel_to_seg}' channel.")    
    with open(file_path,'r') as fp:
        meta = json.load(fp)
    model_settings = meta['segmentation']['cellpose_seg'][channel_to_seg]['model_settings']
    cellpose_eval = meta['segmentation']['cellpose_seg'][channel_to_seg]['cellpose_eval']
    return model_settings,cellpose_eval 

def unpack_kwargs(kwargs: dict, channel_seg: str)-> tuple[str,list[str],dict]:
    """Function to unpack the kwargs and extract necessary variable."""
    if not kwargs:
        nuclear_marker = ""
        channels = [channel_seg]
        metadata = {'finterval':None, 'um_per_pixel':None}
        return nuclear_marker,channels,metadata
    
    if "nuclear_marker" in kwargs:
        nuclear_marker = kwargs['nuclear_marker']
        channels = [channel_seg,nuclear_marker]
        del kwargs['nuclear_marker']

    if 'metadata' in kwargs:
        metadata = kwargs['metadata']
        del kwargs['metadata']
    return nuclear_marker,channels,metadata
    
        

#################### Test ####################
if __name__ == "__main__":
    from os import listdir
    from os.path import join
    
    folder = '/home/Test_images/bigy/HEKA_c1031_c1829_miniSOG_80%_435_2min_40min_002_Merged_s1/Images'
    img_paths = [join(folder,file) for file in sorted(listdir(folder)) if file.endswith('.tif')]
    
    channel = "RFP"
    model_type = "cyto2" #cyto2_cp3, cyto3, /home/Fabian/Models/Cellpose/twoFishMacrophage
    diameter = 30.0
    flow_threshold = 0.5
    cellprob_threshold = 0
    process_as_2D = True
    overwrite = True
    
    cellpose_segmentation(img_paths,channel,model_type,diameter,flow_threshold,cellprob_threshold,overwrite,process_as_2D)


# # # # # # # # main functions # # # # # # # # # 
# def cellpose_segmentation(exp_obj_lst: list[Experiment], channel_seg: str, model_type: str | PathLike ='cyto3',
#                           diameter: float=60., flow_threshold: float=0.4, cellprob_threshold: float=0.0,
#                           overwrite: bool=False, img_fold_src: PathLike="", process_as_2D: bool=False,
#                           save_as_npy: bool=False, nuclear_marker: str="", **kwargs)-> list[Experiment]:
#     """Function to run cellpose segmentation. See https://github.com/MouseLand/cellpose for more details."""

#     for exp_obj in exp_obj_lst:
#         # Activate branch
#         exp_obj.segmentation.is_cellpose_seg = True
#         # Set file type
#         file_type = '.npy' if save_as_npy else '.tif'
#         # Check if exist
#         if is_processed(exp_obj.segmentation.cellpose_seg,channel_seg,overwrite):
#             # Log
#             print(f" --> Cells have already been segmented with cellpose as {file_type} for the '{channel_seg}' channel.")
#             continue
        
#         # Else run cellpose
#         print(f" --> Segmenting cells as {file_type} for the '{channel_seg}' channel")
#         create_save_folder(exp_obj.exp_path,'Masks_Cellpose')
#         delete_old_masks(exp_obj.segmentation.cellpose_seg,channel_seg,exp_obj.cellpose_masks_lst,overwrite)
        
#         # Setup model and eval settings
#         cellpose_setup = CellposeSetup(exp_obj,channel_seg,nuclear_marker,process_as_2D,save_as_npy)
#         cellpose_eval = cellpose_setup.eval_settings(diameter,flow_threshold,cellprob_threshold,**kwargs)
#         model_settings = cellpose_setup.setup_model(model_type)
#         input_data,fold_src = cellpose_setup.gen_input_data(img_fold_src)
        
#         # Run Cellpose
#         parallel_executor(run_cellpose,input_data,model_settings['gpu'],cellpose_eval['z_axis'])
        
#         # Save settings
#         exp_obj.segmentation.cellpose_seg[channel_seg] = {'fold_src':fold_src,'model_settings':model_settings,'cellpose_eval':cellpose_eval}
#         exp_obj.save_as_json()
#     return exp_obj_lst

