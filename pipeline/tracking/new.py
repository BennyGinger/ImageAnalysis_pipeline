from __future__ import annotations
from dataclasses import dataclass, field
from os import PathLike
import os
import os.path as op
from typing import Any
from tifffile import imread
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import cv2
from skimage import io
from skimage.measure import regionprops, regionprops_table
import warnings

from tqdm import trange
warnings.filterwarnings("always")
from pathlib import Path
import re

from pipeline.tracking.gnn_track.modules.resnet_2d.resnet import set_model_architecture, MLP #src_metric_learning.modules.resnet_2d.resnet
from pipeline.utilities.data_utility import run_multithread, run_multiprocess, load_stack, get_img_prop
from skimage.morphology import label

PROPERTIES = ['label', 'area', 'bbox', 'slice', 'centroid', 'major_axis_length', 'minor_axis_length', 'max_intensity', 'mean_intensity', 'min_intensity']

def create_csv(input_images, input_seg, input_model, output_csv, channel:str):
    # Load images
    if not Path(input_images).exists():
        raise FileNotFoundError(f"Input images folder does not exist: {input_images}")
    
    img_lst = sorted([str(file) for file in Path(input_images).glob(f"*{channel}*.tif")])
    
    if not img_lst:
        raise FileNotFoundError(f"Input images folder does not contain any images with the channel: {channel}")
    
    # Load segmentation masks
    if not Path(input_seg).exists():
        raise FileNotFoundError(f"Input segmentation folder does not exist: {input_seg}")
    
    seg_lst = sorted([str(file) for file in Path(input_seg).glob(f"*{channel}*.tif")])
    
    if not seg_lst:
        raise FileNotFoundError(f"Input segmentation folder does not contain any images with the channel: {channel}")
    
    # Create dataset
    ds = ExtractFeatures(images=img_lst, seg_masks=seg_lst, channel=channel)
    
    ds.preprocess_features_loop_by_results_w_metric_learning(path_to_write=output_csv, dict_path=input_model)



@dataclass
class ExtractFeatures():
    """Example dataset class for loading images from folder."""

    img_stack: np.ndarray
    seg_stack: np.ndarray
    # Settings of the model
    model_path: PathLike | str
    model_params: dict[str, Any] = field(init=False)
    model_roi_shape: tuple[int,int] = field(init=False)
    # Regionprops
    props: pd.DataFrame = field(init=False)
    # Properties used to normalize the images
    min_int: int = field(init=False)
    max_int: int = field(init=False)
    # Properties used to pad the images
    curr_roi_shape: tuple[int,int] = field(init=False)
    pad_value: int = field(init=False)
    
    def __post_init__(self):
        # Unpack the model parameters
        self.model_params = torch.load(self.model_path)
        model_roi_shape = self.model_params['roi']
        self.model_roi_shape = (model_roi_shape['row'], model_roi_shape['col'])
        self.pad_value = self.model_params['pad_value']
        
    def get_regionprops(self):
        # Extract the regionprops
        fixed_args = {'mask_array': self.seg_stack,'img_array': self.img_stack}
        lst_df = run_multithread(extract_regionprops,range(self.seg_stack.shape[0]),fixed_args)
        
        # Save the dataframes
        self.props = pd.concat(lst_df, ignore_index=True)
        self.curr_roi_shape = (self.props['delta_row'].max(), self.props['delta_col'].max())
        
        # Get the min and max intensity values
        self.min_int = self.props['min_intensity'].min()
        self.max_int = self.props['max_intensity'].max()
    
    @property
    def ref_shape(self):
        if self.curr_roi_shape[0] > self.model_roi_shape[0]:
            return self.curr_roi_shape
        if self.curr_roi_shape[1] > self.model_roi_shape[1]:
            return self.curr_roi_shape
        return self.model_roi_shape        

def crop_images(img: np.ndarray, mask: np.ndarray, bbox_slice: tuple[slice], mask_idx: int)-> tuple[np.ndarray,np.ndarray]:
    """Function to crop the images and masks to the bbox_slice. The function will return the cropped images and masks as np.arrays. The mask_idx is the value of the mask to crop."""
    
    # Crop the images
    img_crop = img[bbox_slice]
    mask_crop = mask[bbox_slice] == mask_idx
    return img_crop, mask_crop

def normalize_images(img_crop: np.ndarray, mask_crop: np.ndarray, pad_value: int, min_int: int, max_int: int)-> np.ndarray:
    """Return a normalized image with the background set to the pad value. The normalization is done by min-max scaling the intensity values of the image within the mask_crop region. Retruns a np.array as float32."""
    
    
    outter_mask_crop = np.logical_not(mask_crop)
    
    # Set the background to the pad value
    img_crop[outter_mask_crop] = pad_value
    
    # Normalize the intensity values
    img_crop = img_crop.astype(np.float32)
    img_crop[mask_crop] = (img_crop[mask_crop] - min_int) / (max_int - min_int)
    return img_crop
    
def pad_images(img: np.ndarray, ref_shape: tuple[int,int], pad_value: int, output_shape: tuple[int,int])-> np.ndarray:
    """Add padding to the image to match the reference shape. The padding is added to the top and left side of the image. The image is then resized to the output_shape. The function returns the padded and resized image as a np.array. ref_shape and output_shape are in the format (y,x) and ref_shape is either equal or bigger than output_shape."""
    
    # Pad the image and masks, if necessary
    if img.shape[0] != ref_shape[0] or img.shape[1] != ref_shape[1]:
        delta_row = ref_shape[0] - img.shape[0]
        delta_col = ref_shape[1] - img.shape[1]
        pad_top = delta_row // 2
        pad_left = delta_row - pad_top
        # Pad the image and masks, if necessary
        img = cv2.copyMakeBorder(img, pad_top, delta_row - pad_top, pad_left, delta_col - pad_left,cv2.BORDER_CONSTANT, value=pad_value)
    # cv2 inverse the shape i.e. (x,y)
    return cv2.resize(img, dsize=output_shape[::-1])
        
def extract_regionprops(frame_idx: int | None, mask_array: np.ndarray, img_array: np.ndarray, properties: list[str]=None)-> pd.DataFrame:
        """Function to extract the regionprops from the mask_array and img_array. The function will extract the properties defined in the PROPERTIES list. If the ref_masks and/or the sec_maks are provided, the function will extract the dmap from the reference masks and/or whether the cells in pramary masks overlap with cells of the secondary masks. The extracted data will be returned as a pandas.DataFrame.
        
        Args:
            frame (int): The frame index to extract the data from.
            
            mask_array ([[F],Y,X], np.ndarray): The mask array to extract the regionprops from. Frame dim is optional.
            
            img_array ([[F],Y,X,[C]], np.ndarray): The image array to extract the mean intensities from. Frame and channel dim are optional.
            
            properties (list[str], optional): The properties to extract from the regionprops, if different from constant PROPERTIES. Defaults to None.
            
        Returns:
            pd.DataFrame: The extracted regionprops data.
            """
            
        if not properties:
            properties = PROPERTIES
        
        ## Extract the main regionprops
        prop = regionprops_table(mask_array[frame_idx],img_array[frame_idx],properties=properties)
        
        # Get the absolute size of the bbox
        prop['delta_row'] = prop['bbox-2'] - prop['bbox-0']
        prop['delta_col'] = prop['bbox-3'] - prop['bbox-1']
        
        # Create a dataframe
        df = pd.DataFrame(prop)
        df['frame'] = frame_idx
        return df





if __name__ == "__main__":
    from time import time
    
    start = time()
    input_images = '/home/Test_images/nd2/Run4/c4z1t91v1_s1/Images_Registered'
    input_segmentation = '/home/Test_images/nd2/Run4/c4z1t91v1_s1/Masks_Cellpose'
    model = "PhC-C2DH-U373"
    input_model = f"/home/ImageAnalysis_pipeline/pipeline/tracking/gnn_track/models/{model}/all_params.pth"

    output_csv = '/home/Test_images/nd2/Run4/c4z1t91v1_s1/gnn_files'

    create_csv(input_images, input_segmentation, input_model, output_csv, 'RFP')
    end = time()
    print(f"Time to process: {round(end-start,ndigits=3)} sec\n")