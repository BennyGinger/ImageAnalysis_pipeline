from __future__ import annotations
from typing import Callable
import numpy as np
from cellpose import models, core
from cellpose.io import logger_setup, masks_flows_to_seg
from os import PathLike
from os.path import isfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from image_handeling.Experiment_Classes import Experiment
from image_handeling.data_utility import load_stack, is_processed, create_save_folder, delete_old_masks, save_tif, img_list_src

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

def run_cellpose(img_dict: dict)-> None:
    # Load image/stack and model
    img = load_stack(img_dict['imgs_path'],img_dict['channels'],[img_dict['frame']],img_dict['as_2D'])
    model: models.CellposeModel = img_dict['model']
    # Save path
    path: PathLike = img_dict['imgs_path'][0]
    .replace('Images','Masks_Cellpose').replace('Registered','').replace('Blured','')
    # log
    print(f"  ---> Processing img {mask_path}")
    # Run Cellpose
    masks_cp, flows, _ = model.eval(img,**img_dict['cellpose_eval'])
    # Save
    save_mask(img,masks_cp,flows,model.diam_mean,mask_path,img_dict['as_npy'],img_dict['metadata'])

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
        exp_set: Experiment
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
    def __init__(self, exp_set: Experiment, channel_seg: str ,nuclear_marker: str="", process_as_2D: bool=False, save_as_npy: bool=False) -> None:
        self.exp_set = exp_set
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
        if self.exp_set.img_properties.n_slices==1 or self.process_as_2D:
            return self.cellpose_eval
        
        # If 3D
        self.cellpose_eval['z_axis'] = 0
        if self.cellpose_eval['stitch_threshold']==0:
            self.cellpose_eval['do_3D'] = True
            self.cellpose_eval['anisotropy'] = 2.0
        return self.cellpose_eval
    
    def setup_model(self, model_type: str | PathLike ='cyto3', **kwargs)-> dict:
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
            return self.model_settings
        if model_type in IN_HOUSE_MODELS:
            self.model_settings['pretrained_model'] = model_type
            self.model_settings['model_type'] = False
            self.model = models.CellposeModel(**self.model_settings)
            return self.model_settings
        if isfile(model_type):
            self.model_settings['pretrained_model'] = model_type
            self.model_settings['model_type'] = None
            self.model = models.CellposeModel(**self.model_settings)
            return self.model_settings
        # Else raise error
        raise ValueError(" ".join([f"Model type '{model_type}' not recognized.",
                                    f"Please choose one of the following: {BUILD_IN_MODELS}",
                                    "or provide a path to a pretrained model."]))
        
    def gen_input_data(self, img_fold_src: str="")-> list[dict]:
        # Get the list of channel to segment
        channels = [self.channel_seg]
        if self.nuclear_marker:
            channels.append(self.nuclear_marker)
        
        # Sort images by frames and channels
        sorted_frames = {frame:[img for img in img_list_src(self.exp_set,img_fold_src) if f"_f{frame+1:04d}" in img] 
                         for frame in range(self.exp_set.img_properties.n_frames)}
        
        # Generate input data
        input_data = [{'imgs_path':sorted_frames[frame],
                       'frame':frame,
                       'model':self.model,
                       'mask_name':f'{self.channel_seg}_s{self.exp_set.img_properties.n_series:02d}_f{frame+1:04d}_z0001.tif',
                       'cellpose_eval':self.cellpose_eval,
                       'as_2D':self.process_as_2D,
                       'as_npy':self.save_as_npy,
                       'metadata':{'um_per_pixel':self.exp_set.analysis.um_per_pixel,
                               'finterval':self.exp_set.analysis.interval_sec}} 
                      for frame in range(self.exp_set.img_properties.n_frames)]
        
        return input_data

def parallel_executor(func: Callable, input_args: list[dict], gpu: bool, z_axis: int)-> None:
    if gpu and z_axis==None: # If GPU and 2D images: parallelization
        with ThreadPoolExecutor() as executor:
            executor.map(func,input_args)
        return
    
    if gpu and z_axis==0: # If GPU and 3D images: no parallelization
        for args in input_args:
            func(args)
        return
    
    # If CPU 2D or 3D: parallelization
    with ProcessPoolExecutor() as executor:
        executor.map(func,input_args)

# # # # # # # # main functions # # # # # # # # # 
def cellpose_segmentation(exp_set_list: list[Experiment], channel_seg: str, model_type: str | PathLike ='cyto3',
                          diameter: float=60., flow_threshold: float=0.4, cellprob_threshold: float=0.0,
                          overwrite: bool=False, img_fold_src: PathLike="", process_as_2D: bool=False,
                          save_as_npy: bool=False, nuclear_marker: str="", **kwargs)-> list[Experiment]:
    """Function to run cellpose segmentation. See https://github.com/MouseLand/cellpose for more details."""
    
    
    for exp_set in exp_set_list:
        file_type = '.tif'
        if save_as_npy:
            file_type = '.npy'
        
        # Check if exist
        if is_processed(exp_set.masks.cellpose_seg,channel_seg,overwrite):
                # Log
            print(f" --> Cells have already been segmented with cellpose as {file_type} for the '{channel_seg}' channel.")
            continue
        
        # Else run cellpose
        print(f" --> Segmenting cells as {file_type} for the '{channel_seg}' channel")
        create_save_folder(exp_set.exp_path,'Masks_Cellpose')
        delete_old_masks(exp_set.masks.cellpose_seg,channel_seg,exp_set.cellpose_masks_lst,overwrite)
        
        # Setup model and eval settings
        cellpose_setup = CellposeSetup(exp_set,channel_seg,nuclear_marker,process_as_2D,save_as_npy)
        cellpose_eval = cellpose_setup.eval_settings(diameter,flow_threshold,cellprob_threshold,**kwargs)
        model_settings = cellpose_setup.setup_model(model_type)
        input_data = cellpose_setup.gen_input_data(img_fold_src)
        
        # Run Cellpose
        parallel_executor(run_cellpose,input_data,model_settings['gpu'],cellpose_eval['z_axis'])
        
        # Save settings
        exp_set.masks.cellpose_seg[channel_seg] = {'model_settings':model_settings,'cellpose_eval':cellpose_eval}
        exp_set.save_as_json()
    return exp_set_list

