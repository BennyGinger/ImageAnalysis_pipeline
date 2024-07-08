from __future__ import annotations
from dataclasses import dataclass, field
from pipeline.utilities.pipeline_utility import PathType
from typing import Any
import torch
import numpy as np
import pandas as pd
import cv2
from skimage.measure import regionprops_table
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pipeline.tracking.gnn_track.modules.resnet_2d.resnet import set_model_architecture as set_model_arch_2d
from pipeline.tracking.gnn_track.modules.resnet_2d.resnet import MLP as MLP_2D
from pipeline.tracking.gnn_track.modules.resnet_3d.resnet import set_model_architecture as set_model_arch_3d
from pipeline.tracking.gnn_track.modules.resnet_3d.resnet import MLP as MLP_3D
from pipeline.utilities.data_utility import run_multithread, load_stack, get_exp_props
from pipeline.utilities.pipeline_utility import progress_bar


PROPERTIES = ['label', 'area', 'bbox', 'centroid', 'major_axis_length', 'minor_axis_length', 'max_intensity', 'mean_intensity', 'min_intensity']

# [ ]: Test 3D.
############################ Main function ############################
def extract_img_features(img_paths: Path, seg_paths: Path, model_path: Path, save_dir: Path, channel: str, overwrite: bool)-> None:
    """Main function to extract the image features using the metric learning model. 
    
    Args:
        img_fold_src (Path): The path to the folder containing the images.
        
        seg_fold_src (Path): The path to the folder containing the segmentation masks.
        
        model_param_path (Path): The path to the model parameters file.
        
        save_dir (Path): The path to the folder to save the extracted features.
        
        channel (str): The channel to extract the features from."""
        
        
    df_feat_path = save_dir.joinpath('df_feat.csv')
    
    if df_feat_path.exists() and not overwrite:
        # Log
        print(f" --> Image features have already been extracted for the '{channel}' channel")
        return
    
    # Load the images and masks
    exp_path = img_paths.parent
    img_lst = sorted([str(file) for file in img_paths.glob(f"*{channel}*.tif")])
    seg_lst = sorted([str(file) for file in seg_paths.glob(f"*{channel}*.tif")])
    _, _, n_frames, z_slices = get_exp_props(img_lst)
    img_stack = load_stack(img_lst,channel,range(n_frames))
    seg_stack = load_stack(seg_lst,channel,range(n_frames))
    
    
    # Get the roi shape and pad value
    model_params: dict = torch.load(model_path)
    if z_slices > 1:
        model_roi_shape = (model_params['roi']['depth'], model_params['roi']['row'], model_params['roi']['col'])
    else:
        model_roi_shape = (model_params['roi']['row'], model_params['roi']['col'])
    pad_value = model_params['pad_value']
    
    # get all padded images
    print(f"  ---> Preprocessing images for \033[94m{exp_path}\033[0m")
    img_frames = ImagesPreProcessing(img_stack, seg_stack, model_roi_shape).preprocess_frame(pad_value)
    
    # Extract features
    print(f"  ---> Extracting image features")
    trunk, embedder = initialize_models(model_params,z_slices)
    fixed_args = {'trunk': trunk, 'embedder': embedder, 'img_frames': img_frames}
    
    dfs = [_extract_feat(frame_idx, **fixed_args) for frame_idx in progress_bar(range(n_frames))]
    df = pd.concat(dfs, axis=0)
    df.to_csv(df_feat_path, index=False)
        



############################ Helper classes and functions ############################
@dataclass
class ImagesPreProcessing():
    img_stack: np.ndarray # 2D (t,y,x) or 3D (t,z,y,x) image stack
    seg_stack: np.ndarray
    model_roi_shape: tuple[int,int] | tuple[int,int,int] # (y,x) or (z,y,x)
    is_3D: bool = field(init=False)
    curr_roi_shape: tuple[int,int] | tuple[int,int,int] = field(init=False)
    min_max_int: tuple[int,int] = field(init=False)
    img_frames: list[FramesPreProcessing] = field(init=False)

    def __post_init__(self)-> None:
        # Check if the images are 3D
        self.is_3D = True if self.img_stack.ndim > 3 else False

    def get_regionprops(self)-> list[pd.DataFrame]:
        """Extract the regionprops from the segmentation masks. The regionprops will be extracted for each frame in the dataset. The extracted data will be stored in a list of pandas.DataFrames. The min and max intensity values of the whole stack will be stored as 
        to normalize the images. The largest roi shape will be stored as the current shape of the dataset, to be used for padding the images. Retruns a list of props as pd.DataFrames."""
        
        # Extract the regionprops
        fixed_args = {'mask_array': self.seg_stack,'img_array': self.img_stack, 'is_3D': self.is_3D}
        lst_df = run_multithread(extract_regionprops,range(self.img_stack.shape[0]),fixed_args)
        
        # Convert to a single DataFrame and extract the max roi shape
        props = pd.concat(lst_df, ignore_index=True)
        if self.is_3D:
            delta_depth = props['max_depth_bb'] - props['min_depth_bb']
            delta_row = props['max_row_bb'] - props['min_row_bb']
            delta_col = props['max_col_bb'] - props['min_col_bb']
            self.curr_roi_shape = (delta_depth.max(), delta_row.max(), delta_col.max())
        else:
            delta_row = props['max_row_bb'] - props['min_row_bb']
            delta_col = props['max_col_bb'] - props['min_col_bb']
            self.curr_roi_shape = (delta_row.max(), delta_col.max())
        
        # Get the min and max intensity values
        self.min_max_int = (props['min_intensity'].min(),props['max_intensity'].max())
        return lst_df
    
    def preprocess_frame(self, pad_value: int)-> list[FramesPreProcessing]:
        """Preprocess the images in the dataset. The images will be fractionated into the regions of interest (i.e. masks). Then they will be padded to the reference shape and resized to the model shape. The images will be normalized using the min_max_int values. The images will be stored in the img_frames list."""
        
        # Get the regionprops
        print("    ----> Extracting regionprops")
        lst_df = self.get_regionprops()
        
        # Preprocess the frames
        print("    ----> Preprocessing frames")
        self.img_frames = []
        for frame_idx, frame_props in enumerate(lst_df):
            frame = FramesPreProcessing(frame_props, self.img_stack[frame_idx], self.seg_stack[frame_idx])
            frame.pad_images(pad_value, self.min_max_int, self.ref_shape, self.model_roi_shape)
            self.img_frames.append(frame)
        return self.img_frames
    
    @property
    def ref_shape(self):
        """Get the reference shape for padding the images. The reference shape is the maximum shape of the images in the dataset. If the current shape is bigger than the model shape, the current shape will be returned. Otherwise, the model shape will be returned."""
        
        
        if self.curr_roi_shape[0] > self.model_roi_shape[0]:
            return self.curr_roi_shape
        if self.curr_roi_shape[1] > self.model_roi_shape[1]:
            return self.curr_roi_shape
        return self.model_roi_shape       


@dataclass
class FramesPreProcessing():
    frame_props: pd.DataFrame
    img_frame: np.ndarray # 2D (y,x) or 3D (z,y,x) image stack
    seg_frame: np.ndarray
    is_3D: bool = field(init=False)
    bbox_slices: list[tuple[slice,slice] | tuple[slice,slice,slice]] = field(default_factory=list)
    padded_images: list[np.ndarray] = field(init=False)
    
    def __post_init__(self)-> None:
        self.is_3D = True if self.img_frame.ndim > 2 else False
        
        # Construct the bbox slices
        if self.is_3D:
            for _, row in self.frame_props.iterrows():
                depth_slice = slice(int(row['min_depth_bb']), int(row['max_depth_bb']))
                row_slice = slice(int(row['min_row_bb']), int(row['max_row_bb']))
                col_slice = slice(int(row['min_col_bb']), int(row['max_col_bb']))
                
                self.bbox_slices.append((depth_slice, row_slice, col_slice))
        else:
            for _, row in self.frame_props.iterrows():
                row_slice = slice(int(row['min_row_bb']), int(row['max_row_bb']))
                col_slice = slice(int(row['min_col_bb']), int(row['max_col_bb']))
                self.bbox_slices.append((row_slice, col_slice))
    
    def pad_images(self, pad_value: int, min_max_int: tuple[int,int], ref_shape: tuple[int,int] | tuple[slice,slice,slice], output_shape: tuple[int,int] | tuple[slice,slice,slice])-> None:
        
        self.padded_images = []
        fixed_args = {'img_frame': self.img_frame, 'seg_frame': self.seg_frame, 'bbox_slices': self.bbox_slices, 'pad_value': pad_value, 'min_max_int': min_max_int, 'ref_shape': ref_shape, 'output_shape': output_shape,'is_3D': self.is_3D}
        
        with ThreadPoolExecutor() as executor:
            self.padded_images = list(executor.map(partial(self._pad_images, **fixed_args), range(len(self.bbox_slices))))
            
    @staticmethod
    def _pad_images(mask_idx: int, img_frame: np.ndarray, seg_frame: np.ndarray, bbox_slices: list[tuple[slice,slice]], pad_value: int, min_max_int: tuple[int,int], ref_shape: tuple[int,int] | tuple[slice,slice,slice], output_shape: tuple[int,int] | tuple[slice,slice,slice], is_3D: bool)-> np.ndarray:
        
        bbox_slice = bbox_slices[mask_idx]
        img_crop, mask_crop = crop_images(img_frame, seg_frame, bbox_slice, mask_idx+1)
        img_norm = normalize_images(img_crop, mask_crop, pad_value, *min_max_int)
        if is_3D:
            return pad_and_resize_image_3D(img_norm, ref_shape, pad_value, output_shape)
        return pad_and_resize_image_2D(img_norm, ref_shape, pad_value, output_shape)


def crop_images(img: np.ndarray, mask: np.ndarray, bbox_slice: tuple[slice,slice] | tuple[slice,slice,slice], mask_idx: int)-> tuple[np.ndarray,np.ndarray]:
    """Function to crop the images and masks to the bbox_slice. The function will return the cropped images and masks as np.arrays. The mask_idx is the value of the mask to crop."""
    img_crop = img.copy() ; mask_crop = mask.copy()
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
    
def pad_and_resize_image_2D(img: np.ndarray, ref_shape: tuple[int,int] | tuple[slice,slice,slice], pad_value: int, output_shape: tuple[int,int] | tuple[slice,slice,slice])-> np.ndarray:
    """Add padding to the image to match the reference shape. The padding is added to the top and left side of the image. The image is then resized to the output_shape. The function returns the padded and resized image as a np.array. ref_shape and output_shape are in the format (y,x) and ref_shape is either equal or bigger than output_shape."""
    
    # Pad the image and masks, if necessary
    if img.shape[0] != ref_shape[0] or img.shape[1] != ref_shape[1]:
        delta_row = ref_shape[0] - img.shape[0]
        delta_col = ref_shape[1] - img.shape[1]
        pad_top = delta_row // 2
        pad_left = delta_col // 2
        # Pad the image and masks, if necessary
        img = cv2.copyMakeBorder(img, pad_top, delta_row - pad_top, pad_left, delta_col - pad_left,cv2.BORDER_CONSTANT, value=pad_value)
    # cv2 inverse the shape i.e. (x,y)
    return cv2.resize(img, dsize=output_shape[::-1])
     
def pad_and_resize_image_3D(img: np.ndarray, ref_shape: tuple[int,int] | tuple[slice,slice,slice], pad_value: int, output_shape: tuple[int,int] | tuple[slice,slice,slice])-> np.ndarray:
    """Add padding to the image to match the reference shape. The padding is added to the top and left side of the image. The image is then resized to the output_shape. The function returns the padded and resized image as a np.array. ref_shape and output_shape are in the format (y,x) and ref_shape is either equal or bigger than output_shape."""
    
    # Pad the image and masks, if necessary
    if img.shape[0] != ref_shape[0] or img.shape[1] != ref_shape[1] or img.shape[2] != ref_shape[2]:
        delta_depth = ref_shape[0] - img.shape[0]
        delta_row = ref_shape[1] - img.shape[1]
        delta_col = ref_shape[2] - img.shape[2]
        
        pad_depth = delta_depth // 2
        pad_top = delta_row // 2
        pad_left = delta_col // 2
        # Pad the image and masks, if necessary
        img = np.pad(img, 
                     pad_width=((pad_depth, delta_depth - pad_depth), 
                                (pad_top, delta_row - pad_top),
                                (pad_left, delta_col - pad_left)),
                     mode='constant',
                     constant_values=np.ones((3, 2)) * pad_value)
    # Interpolate the resized image
    img = torch.nn.functional.interpolate(torch.from_numpy(img[None, None, ...]),
                                                    size=output_shape,
                                                    mode='trilinear')
    return img.numpy().squeeze()
        
def extract_regionprops(frame_idx: int | None, mask_array: np.ndarray, img_array: np.ndarray, is_3D: bool, properties: list[str]=None, **kwargs)-> pd.DataFrame:
        """Function to extract the regionprops from the mask_array and img_array. The function will extract the properties defined in the PROPERTIES list. If the ref_masks and/or the sec_maks are provided, the function will extract the dmap from the reference masks and/or whether the cells in pramary masks overlap with cells of the secondary masks. The extracted data will be returned as a pandas.DataFrame."""
            
        if not properties:
            properties = PROPERTIES
        
        ## Extract the main regionprops
        prop = regionprops_table(mask_array[frame_idx],img_array[frame_idx],properties=properties)
        
        # Create a dataframe
        df = pd.DataFrame(prop)
        
        # Rename the columns
        if is_3D:
            col_rename = {'label': 'seg_label',
                          'bbox-0': 'min_depth_bb',
                          'bbox-1': 'min_row_bb',
                          'bbox-2': 'min_col_bb',
                          'bbox-3': 'max_depth_bb',
                          'bbox-4': 'max_row_bb',
                          'bbox-5': 'max_col_bb',
                          'centroid-0': 'centroid_depth',
                          'centroid-1': 'centroid_row',
                          'centroid-2': 'centroid_col',}
        else:
            col_rename = {'label': 'seg_label',
                    'bbox-0': 'min_row_bb',
                    'bbox-1': 'min_col_bb',
                    'bbox-2': 'max_row_bb',
                    'bbox-3': 'max_col_bb',
                    'centroid-0': 'centroid_row',
                    'centroid-1': 'centroid_col'}
        df.rename(columns=col_rename, inplace=True)
        return df

def initialize_models(model_params: dict[str, Any], z_slices: int)-> tuple[torch.nn.Module, torch.nn.Module]:
    # Import the model
    if z_slices > 1:
        trunk = set_model_arch_3d(model_params['model_name'])
        embedder = MLP_3D(model_params['mlp_dims'], normalized_feat=model_params['mlp_normalized_features'])
    else:
        trunk = set_model_arch_2d(model_params['model_name'])
        embedder = MLP_2D(model_params['mlp_dims'], normalized_feat=model_params['mlp_normalized_features'])
    
    # Initialize trunk
    trunk.load_state_dict(model_params['trunk_state_dict'])
    trunk.eval()
    # Initialize embedder
    embedder.load_state_dict(model_params['embedder_state_dict'])
    embedder.eval()
    return trunk, embedder

def extract_freature_metric_learning(padded_images: list[np.ndarray], trunk: torch.nn.Module, embedder: torch.nn.Module)-> np.ndarray:
    """Extract features from a frame using the metric learning model. The function will return the embedded image as a np.array."""
    
    # Convert the images to torch.Tensor
    padded_tensor = torch.stack([torch.from_numpy(img).float() for img in padded_images])
    
    with torch.no_grad():
        embedded_img = embedder(trunk(padded_tensor[:, None, ...]))

    return embedded_img.numpy().squeeze()

def _extract_feat(frame_idx: int, trunk: torch.nn.Module, embedder: torch.nn.Module, img_frames: list[FramesPreProcessing])-> pd.DataFrame:
    
    frame = img_frames[frame_idx]
    
    embedded_array = extract_freature_metric_learning(frame.padded_images, trunk, embedder)
    # Construct csv
    return construct_csv(frame_idx, frame, embedded_array)

def construct_csv(frame_idx: int, frame: FramesPreProcessing, embedded_array: np.ndarray)-> pd.DataFrame:
    
    feat_cols = [f'feat_{i}' for i in range(embedded_array.shape[1])]
    embedded_df = pd.DataFrame(embedded_array, columns=feat_cols)
    props_df = frame.frame_props
    df = pd.concat([props_df, embedded_df], axis=1)
    df['frame_num'] = frame_idx
    
    # Save the data
    # save_path = Path(output_path).joinpath("csv").joinpath(f"frame_{frame_idx:04d}.csv")
    # save_path.parent.mkdir(exist_ok=True, parents=True)
    # df.to_csv(save_path, index=False)
    return df



if __name__ == "__main__":
    from time import time
    
    start = time()
    input_images = '/home/Test_images/nd2/Run4/c4z1t91v1_s1/Images_Registered'
    input_segmentation = '/home/Test_images/nd2/Run4/c4z1t91v1_s1/Masks_Cellpose'
    model = "PhC-C2DH-U373"
    input_model = f"/home/ImageAnalysis_pipeline/pipeline/tracking/gnn_track/models/{model}/all_params.pth"

    output_csv = '/home/Test_images/nd2/Run4/c4z1t91v1_s1/gnn_files'

    extract_img_features(input_images, input_segmentation, input_model, output_csv, 'RFP')
    end = time()
    print(f"Time to process: {round(end-start,ndigits=3)} sec\n")