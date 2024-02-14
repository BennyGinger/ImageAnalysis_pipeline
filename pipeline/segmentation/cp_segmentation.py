from __future__ import annotations
from typing import Callable
import numpy as np
from cellpose import models, core
from cellpose.io import logger_setup, masks_flows_to_seg
from os import PathLike
from os.path import isfile
from tifffile import imsave
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from Experiment_Classes import Experiment
from loading_data import load_stack, is_processed, create_save_folder, gen_input_data, delete_old_masks

MODEL_SETTINGS = {'gpu':core.use_gpu(),
                  'model_type': 'cyto2',
                  'pretrained_model':False,
                  'net_avg':False,
                  'device':None,
                  'diam_mean':30.,
                  'residual_on':True,
                  'style_on':True,
                  'concatenation':False,
                  'nchan':2}

CELLPOSE_EVAL = {'batch_size':8,
                 'channels':[0,0],
                 'channel_axis':None,
                 'z_axis':None,
                 'invert':False,
                 'normalize':True,
                 'diameter':60.,
                 'do_3D':False,
                 'anisotropy':None,
                 'net_avg':False,
                 'augment':False,
                 'tile':True,
                 'tile_overlap':0.1,
                 'resample':True,
                 'interp':True,
                 'flow_threshold':0.4,
                 'cellprob_threshold':0.,
                 'min_size':500,
                 'stitch_threshold':0.,
                 'rescale':None,
                 'progress':None,
                 'model_loaded':False}

BUILD_IN_MODELS = ['cyto','nuclei','tissuenet','livecell','cyto2',
                     'CP','CPx','TN1','TN2','TN3','LC1','LC2','LC3','LC4']

IN_HOUSE_MODELS = []

def apply_cellpose_segmentation(img_dict: dict)-> None:
    img = load_stack(img_dict['imgs_path'],img_dict['channel_seg_list'],[img_dict['frame']])
    if img_dict['as_2D'] and img.ndim==3:
        img = np.amax(img,axis=0)
    print(f"  ---> Processing frame {img_dict['frame']+1}")
    mask_path = img_dict['imgs_path'][0].replace("Images","Masks_Cellpose").replace('_Registered','').replace('_Blured','')
    
    # Run Cellpose. Returns 4 variables
    if img_dict['as_npy']:
        masks_cp, flows, _ = img_dict['model'].eval(img,**img_dict['cellpose_eval'])
        save_npy(img,masks_cp,flows,img_dict['model'].diam_mean,mask_path)
        return
    
    masks_cp, __, __, = img_dict['model'].eval(img,**img_dict['cellpose_eval'])
    save_tiff(masks_cp,mask_path)

def save_npy(img: np.ndarray | list[np.ndarray], masks_cp: np.ndarray | list[np.ndarray], flows: list[np.ndarray] | list[list], 
             diameter: float, mask_path: str)-> None:
    if img.ndim==3:
        mask_path = mask_path.replace("_z0001","_allz")
    masks_flows_to_seg(img,masks_cp,flows,diameter,mask_path)

def save_tiff(masks_cp: np.ndarray | list[np.ndarray], mask_path: str)-> None:
    if masks_cp.ndim==3:
        for z_silce in range(masks_cp.shape[0]):
            mask_path = mask_path.replace("_z0001",f"_z{z_silce+1:04d}")
            imsave(mask_path,masks_cp[z_silce,...].astype('uint16'))
    else:
        imsave(mask_path,masks_cp.astype('uint16'))
                     
def initialize_model(model_settings: dict)-> tuple[models.CellposeModel,dict]:
    logger_setup()
    if model_settings['model_type'] in IN_HOUSE_MODELS:
        model_settings['pretrained_model'] = model_settings['model_type']
        model_settings['model_type'] = False
        model = models.CellposeModel(**model_settings)
        return model,model_settings
    
    if model_settings['model_type'] in BUILD_IN_MODELS:
        model = models.CellposeModel(**model_settings)
        return model,model_settings
    
    if isfile(model_settings['model_type']):
        model_settings['pretrained_model'] = model_settings['model_type']
        model_settings['model_type'] = None
        model = models.CellposeModel(**model_settings)
    else:
        raise ValueError(" ".join([f"Model type '{model_settings['model_type']}' not recognized.",
                                    f"Please choose one of the following: {BUILD_IN_MODELS}",
                                    "or provide a path to a pretrained model."]))
    return model,model_settings

def setup_cellpose_eval(cellpose_eval: dict, n_slices: int, as_2D: bool, nuclear_marker: str="")-> dict:
    if nuclear_marker:
        cellpose_eval['channels'] = [1,2]
    
    # If 2D or 3D to be treated as 2D
    if n_slices==1 or as_2D:
        return cellpose_eval
    
    # If 3D
    cellpose_eval['z_axis'] = 0
    cellpose_eval['do_3D'] = True
    cellpose_eval['anisotropy'] = 2.0
    if cellpose_eval['stitch_threshold']>0:
        cellpose_eval['anisotropy'] = None
        cellpose_eval['do_3D'] = False
    
    return cellpose_eval

def initialize_cellpose(n_slices: int, as_2D: bool, model_type: str | PathLike ='cyto2', diameter: float=60., nuclear_marker: str="", 
                             flow_threshold: float=0.4, cellprob_threshold: float=0.0, **kwargs)-> tuple[models.CellposeModel,dict,dict]:
    # Default settings for cellpose model
    model_settings = MODEL_SETTINGS; cellpose_eval = CELLPOSE_EVAL
    cellpose_eval['diameter'] = diameter; cellpose_eval['flow_threshold'] = flow_threshold; cellpose_eval['cellprob_threshold'] = cellprob_threshold
    model_settings['model_type'] = model_type
    
    # Unpack kwargs
    if not kwargs:
        cellpose_eval = setup_cellpose_eval(cellpose_eval,n_slices,as_2D,nuclear_marker)
        return *initialize_model(model_settings),cellpose_eval
    
    for k,v in kwargs.items():
        if k in model_settings:
            model_settings[k] = v
        elif k in cellpose_eval:
            cellpose_eval[k] = v
    cellpose_eval = setup_cellpose_eval(cellpose_eval,n_slices,as_2D,nuclear_marker)
    return *initialize_model(model_settings),cellpose_eval

def parallel_executor(func: Callable, input_args: list, gpu: bool)-> None:
    if gpu and len(input_args[0]['imgs_path'])==1: # If GPU and 2D images: parallelization
        with ThreadPoolExecutor() as executor:
            executor.map(func,input_args)
    elif gpu and len(input_args[0]['imgs_path'])>1: # If GPU and 3D images: no parallelization
        for args in input_args:
            func(args)
    else: # If CPU 2D or 3D: parallelization
        with ProcessPoolExecutor() as executor:
            executor.map(func,input_args)

# # # # # # # # main functions # # # # # # # # # 
def cellpose_segmentation(exp_set_list: list[Experiment], channel_to_seg: str, model_type: str | PathLike ='cyto2',
                          diameter: float=60., flow_threshold: float=0.4, cellprob_threshold: float=0.0,
                          cellpose_overwrite: bool=False, img_fold_src: str="", as_2D: bool=False,
                          as_npy: bool=False, nuclear_marker: str="", **kwargs)-> list[Experiment]:
    """Function to run cellpose segmentation. See https://github.com/MouseLand/cellpose for more details."""
    
    
    for exp_set in exp_set_list:
        file_type = '.tif'
        if as_npy:
            file_type = '.npy'
        
        # Check if exist
        if is_processed(exp_set.masks.cellpose_seg,channel_to_seg,cellpose_overwrite):
                # Log
            print(f" --> Cells have already been segmented with cellpose as {file_type} for the '{channel_to_seg}' channel.")
            continue
        
        # Else run cellpose
        print(f" --> Segmenting cells as {file_type} for the '{channel_to_seg}' channel")
        create_save_folder(exp_set.exp_path,'Masks_Cellpose')
        delete_old_masks(exp_set.masks.cellpose_seg,channel_to_seg,exp_set.mask_cellpose_list,cellpose_overwrite)
        
        # Setup model and eval settings
        model,model_settings,cellpose_eval = initialize_cellpose(exp_set.img_properties.n_slices,as_2D,
                                                                model_type,diameter,flow_threshold,cellprob_threshold,**kwargs)
        cellpose_channels = [channel_to_seg]
        if nuclear_marker: cellpose_channels.append(nuclear_marker)
        
        # Generate input data
        img_data = gen_input_data(exp_set,img_fold_src,cellpose_channels,model=model,cellpose_eval=cellpose_eval,as_2D=as_2D,as_npy=as_npy)
        
        # Cellpose
        parallel_executor(apply_cellpose_segmentation,img_data,model_settings['gpu'])
        
        # Save settings
        exp_set.masks.cellpose_seg[channel_to_seg] = {'model_settings':model_settings,'cellpose_eval':cellpose_eval}
        exp_set.save_as_json()
    return exp_set_list
