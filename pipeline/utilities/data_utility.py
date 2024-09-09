from __future__ import annotations
from os import remove
from os.path import join
from pathlib import Path
from pipeline.utilities.pipeline_utility import progress_bar, PathType
from pipeline.utilities.Experiment_Classes import Experiment
from typing import Iterable, Iterator, Callable
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
from threading import Lock
import numpy as np
from tifffile import imwrite
# NOTE: Added imageio, as I have png mask file as well, from cellpose manual segmentation. Should be deleted in the future
from imageio import imread


def _load_images(img_paths: list[Path], channel: str, frame: int) -> list:
    """Load images for a given channel and frame."""
    return [imread(path) for path in img_paths if path.match(f"*{channel}*_f{frame+1:04}*")]

def _load_frames(img_paths: list[Path], channel: str, frame_range: Iterable[int]) -> list:
    """Load all frames for a given channel."""
    return [_load_images(img_paths, channel, frame) for frame in frame_range]

def _load_channels(img_paths: list[Path], channels: Iterable[str], frame_range: Iterable[int]) -> list:
    """Load all channels."""
    return [_load_frames(img_paths, channel, frame_range) for channel in channels]

def _convert_to_stack(exp_list: list, channels: Iterable[str]) -> np.ndarray:
    """Process the stack."""
    if len(channels) == 1:
        return np.squeeze(np.stack(exp_list))
    else:
        return np.moveaxis(np.squeeze(np.stack(exp_list)), [0], [-1])

def _prepare_imgs(img_paths: list[PathType | Path], channels: str | Iterable[str] | None, frame_range: int | Iterable[int] | None=None)-> tuple[list[Path],list[str],Iterable[int]]:
    if frame_range is None:
        frames = get_exp_props(img_paths)[2]
        frame_range = range(frames)
    
    if channels is None:
        channels = get_exp_props(img_paths)[0]
    
    # Convert to list if string or int
    channels = [channels] if isinstance(channels, str) else channels
    frame_range = [frame_range] if isinstance(frame_range, int) else frame_range

    # check if channels are in the img_paths
    for channel in channels:
        if not is_channel_in_lst(channel, img_paths):
            raise ValueError(f"Channel '{channel}' not found in the image paths")
    
    # Convert img_path to Path object
    img_paths = [Path(path) for path in img_paths]
    return img_paths, channels, frame_range

def load_stack(img_paths: list[PathType | Path], channels: str | Iterable[str] | None=None, frame_range: int | Iterable[int] | None=None, return_2D: bool=False) -> np.ndarray:
    """Convert images to stack. If return_2D is True, return the max projection of the stack. The output shape is tzxyc,
    with t, z and c being optional."""
    
    # Prepare images
    img_paths, channels, frame_range = _prepare_imgs(img_paths, channels, frame_range)
    
    # Load/Reload stack. Expected shape of images tzxyc
    exp_list = _load_channels(img_paths, channels, frame_range)

    # Process stack
    stack = _convert_to_stack(exp_list, channels)

    # If stack is already 2D or want to load 3D, i.e. if frame list contains only one img
    if len(exp_list[0][0]) == 1 or not return_2D:
        return stack

    # For maxIP, if stack is time series, then z is in axis 1
    if len(frame_range) > 1:
        return np.amax(stack, axis=1)
    # if not then z is axis 0
    else:
        return np.amax(stack, axis=0)

def img_list_src(exp_set: Experiment, img_fold_src: str)-> tuple[str,list[PathType]]:
    """If not manually specified, return the latest processed images list
    Return the image folder source to save during segmentation.
    Args:
        exp_set (Experiment): The experiment settings.
        img_fold_src (str): The image folder source. Can be None or str.
    Returns:
        tuple[str,list[PathType]]: The image folder source and the list of masks."""
    
    if img_fold_src and img_fold_src == 'Images':
        return img_fold_src, exp_set.ori_imgs_lst
    if img_fold_src and img_fold_src == 'Images_Registered':
        return img_fold_src, exp_set.registered_imgs_lst
    if img_fold_src and img_fold_src == 'Images_Blured':
        return img_fold_src, exp_set.blured_imgs_lst
    
    # If not manually specified, return the latest processed images list
    if exp_set.preprocess.is_img_blured:
        return 'Images_Blured', exp_set.blured_imgs_lst
    elif exp_set.preprocess.is_frame_reg:
        return 'Images_Registered', exp_set.registered_imgs_lst
    else:
        return 'Images', exp_set.ori_imgs_lst

def seg_mask_lst_src(exp_set: Experiment, mask_fold_src: str)-> tuple[str, list[PathType]]:
    """If not manually specified, return the latest processed segmentated masks list. 
    Return the mask folder source to save during tracking.
    Args:
        exp_set (Experiment): The experiment settings.
        mask_fold_src (str): The mask folder source. Can be None or str.
    Returns:
        tuple[str,list[PathType]]: The mask folder source and the list of masks."""
    
    if mask_fold_src == 'Masks_Threshold':
        return mask_fold_src, exp_set.threshold_masks_lst  
    if mask_fold_src == 'Masks_Cellpose':
        return mask_fold_src, exp_set.cellpose_masks_lst
    
    # If not manually specified, return the latest processed images list
    if exp_set.segmentation.is_threshold_seg:
        return 'Masks_Threshold', exp_set.threshold_masks_lst
    if exp_set.segmentation.is_cellpose_seg:
        return 'Masks_Cellpose', exp_set.cellpose_masks_lst
    else:
        print("No segmentation masks found")

def track_mask_lst_src(exp_set: Experiment, mask_fold_src: str)-> tuple[str, list[PathType]]:
    """If not manually specified, return the latest processed tracked masks list
    Args:
        exp_set (Experiment): The experiment settings.
        mask_fold_src (str): The mask folder source. Can be None or str.
    Returns:
        list[PathType]: The list of tracked masks."""
    
    if mask_fold_src == 'Masks_IoU_Track':
        return mask_fold_src, exp_set.iou_tracked_masks_lst 
    if mask_fold_src == 'Masks_Manual_Track':
        return mask_fold_src, exp_set.man_tracked_masks_lst
    if mask_fold_src == 'Masks_GNN_Track':
        return mask_fold_src, exp_set.gnn_tracked_masks_lst
    
    # If not manually specified, return the latest processed images list
    if exp_set.tracking.manual_tracking:
        return "Masks_Manual_Track", exp_set.man_tracked_masks_lst
    if exp_set.tracking.is_gnn_tracking:
        return "Masks_GNN_Track", exp_set.gnn_tracked_masks_lst
    if exp_set.tracking.iou_tracking:
        return "Masks_IoU_Track", exp_set.iou_tracked_masks_lst
    
    print("No tracking masks found")
    return None, None

def is_processed(process_settings: dict | list, channel_seg: str = None, overwrite: bool = False)-> bool:
    if overwrite:
        return False
    if not process_settings:
        return False
    if isinstance(process_settings, list):
        return True
    if channel_seg not in process_settings:
        return False
    return True

def create_save_folder(exp_path: PathType | Path, folder_name: str)-> PathType:
    if isinstance(exp_path, str):
        exp_path = Path(exp_path)
    save_folder = exp_path.joinpath(folder_name)
    log_path = join(exp_path.stem,folder_name)
    if not save_folder.exists():
        print(f"  ---> Creating saving folder: {log_path}")
        save_folder.mkdir(exist_ok=True)
        return str(save_folder)
    print(f"  ---> Saving folder already exists: {log_path}")
    return str(save_folder)

def delete_old_masks(class_setting_dict: dict, channel_seg: str, mask_files_list: list[PathType], overwrite: bool=False)-> None:
    """Check if old masks exists, if the case, the delete old masks. Only
    if overwrite is True and class_setting_dict is not empty and channel_seg is in class_setting_dict"""
    if not overwrite:
        return
    if not class_setting_dict:
        return
    if channel_seg not in class_setting_dict:
        return
    print(f" ---> Deleting old masks for the '{channel_seg}' channel")
    files_list = [file for file in mask_files_list if file.__contains__(channel_seg)]
    for file in files_list:
        if file.endswith((".tif",".tiff",".npy")):
            remove(file)

def get_resolution(um_per_pixel: tuple[float,float])-> tuple[float,float]:
    x_umpixel,y_umpixel = um_per_pixel
    return 1/x_umpixel,1/y_umpixel

def save_tif(array: np.ndarray, save_path: PathType, um_per_pixel: tuple[float,float], finterval: int, lock: Lock=None)-> None:
    """Save array as tif with metadata. If lock is provided, use it to limit access to the save function."""
    if not lock:
        _save_tif(array,save_path,um_per_pixel,finterval)
        return
    # If lock is provided, use it to limit access to the save function
    with lock:
        _save_tif(array,save_path,um_per_pixel,finterval)

def _save_tif(array: np.ndarray, save_path: PathType, um_per_pixel: tuple[float,float], finterval: int)-> None:
    """Actual save function for tif with metadata. If no metadata provided, save the array as tif without metadata"""
    # If no metadata provided
    if not finterval or not um_per_pixel:
        imwrite(save_path,array.astype(np.uint16))
        return
    # Unpack metadata
    imagej_metadata = {'finterval':finterval, 'unit': 'um'}
    imwrite(save_path,array.astype(np.uint16),imagej=True,metadata=imagej_metadata,resolution=get_resolution(um_per_pixel))

def run_multithread(func: Callable, input_data: Iterable, fixed_args: dict=None)-> list:
    """Run a function in multi-threading. It uses a lock to limit access of some functions to the different threads."""
    if not fixed_args:
        fixed_args = {}
    
    # Run callable in threads
    outputs = []
    with ThreadPoolExecutor() as executor:
        with progress_bar(total=len(input_data)) as pbar:
            # Add lock to fixed_args
            if 'metadata' in fixed_args:
                fixed_args['metadata']['lock'] = Lock()
            else:
                fixed_args['lock'] = Lock()
            # Run function
            results = executor.map(partial(func,**fixed_args),input_data)
            # Update the pbar and get outputs
            for output in results:
                pbar.update()
                outputs.append(output)
    return outputs

def run_multiprocess(func: Callable, input_data: Iterable, fixed_args: dict=None)-> Iterator:
    """Run a function in multi-processing."""
    if not fixed_args:
        fixed_args = {}
    
    # Run cellpose in threads
    with ProcessPoolExecutor() as executor:
        with progress_bar(total=len(input_data)) as pbar:
            results = executor.map(partial(func,**fixed_args),input_data)
            # Update the pbar
            outputs = []
            for output in results:
                pbar.update()
                outputs.append(output)
    return outputs

def get_exp_props(img_paths: list[PathType | Path])-> tuple[list[str],int,int,int]:
    """Function that extract basic properties of the experiment from the image paths. Images names are expected to be in the format: [C]_[s\d{2}]_[f\d{4}]_[z\d{4}] where C is the channel label (any), s\d{2} is the series, f\d{4} is the frame and z\d{4} is the z-slice. \d{2} means followed by 2 digits and \d{4} means by 4 digits.
    
    Returns:
        tuple[list,int,int,int]: The list of channels, the number of series, the number of frames and the number of z-slices."""
    # Convert to Path type
    img_paths = [Path(path) for path in img_paths]
    
    channels = set(); series = set(); frames = set(); z_slices = set()
    for path in img_paths:
        chan, serie, frame, z_slice = path.stem.split('_')
        channels.add(chan)
        series.add(serie)
        frames.add(frame)
        z_slices.add(z_slice)
    return list(channels), len(series), len(frames), len(z_slices)

def is_channel_in_lst(channel: str, img_paths: list[PathType | Path]) -> bool:
    """
    Check if a channel is in the list of image paths.

    Args:
        channel (str): The channel name.
        img_paths (list[PathType]): The list of image paths.

    Returns:
        bool: True if the channel is in the list, False otherwise.
    """
    # Convert to Path type
    img_paths = [Path(path) for path in img_paths]
    return any(channel in path.stem for path in img_paths)


if __name__ == '__main__':
    path = '/home/Test_images/nd2/Run2/c2z25t23v1_nd2_s1/Images_Registered'
    img_lst = list(Path(path).glob('*.tif'))
    # frames, z_slice, channels = get_img_prop(img_lst)
    stack = load_stack(img_lst,frame_range=[5,10])
    print(stack.shape)
