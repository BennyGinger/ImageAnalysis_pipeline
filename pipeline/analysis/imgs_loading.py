from __future__ import annotations
from pipeline.utilities.data_utility import load_stack, get_exp_props
import numpy as np
from scipy.ndimage import distance_transform_edt
from pipeline.mask_transformation.utils import erode_masks
from pathlib import Path
from typing import TypeVar

# Custom variable type
T = TypeVar('T')

def load_images_and_masks(chunk_frames, img_paths, masks_fold, ref_masks_fold, exp_path, pixel_resolution, channels, nframes):
    img_array = load_stack(img_paths, channels, chunk_frames, True)
    
    # Load ref masks, if provided, as ref_masks_fold can be an empty list
    ref_masks = load_reference_masks(chunk_frames, ref_masks_fold, exp_path, pixel_resolution)

    # Load masks
    pair_arrays = generate_mask_pairs(chunk_frames, masks_fold, exp_path)

    # Load diff masks
    if nframes == 1:
        do_diff = False
    diff_array = load_diff_masks(img_array) if do_diff else None
    return img_array, diff_array, ref_masks, pair_arrays

############## Helper Functions ##############
def generate_mask_pairs(chunk_frames: range, masks_fold: list[str], exp_path: Path)-> list[tuple[tuple[np.ndarray, str], list[tuple[np.ndarray, str]] | None]]: 
    pair_arrays: list[tuple[tuple[np.ndarray, str], list[tuple[np.ndarray, str]] | None]] = []
    for fold in masks_fold:
        mask_path = exp_path.joinpath(fold)
        process_name = fold.split('_', maxsplit=1)[-1].lower()
        mask_files = sorted(mask_path.glob('*.tif'))
        mask_channels = get_exp_props(mask_files)[0]
        if mask_channels == 1:
            pair_channels = [(mask_channels[0], None)]
        else:
            pair_channels = make_pairs(mask_channels)
        
        # Erode secondary masks
        for chan, sec_channels in pair_channels:
            primary_mask = load_stack(mask_files, chan, chunk_frames, True)
            primary_name = f"{process_name}_{chan}"
            if sec_channels is not None:
                sec_masks = [load_stack(mask_files, sec_chan, chunk_frames, True) for sec_chan in sec_channels]
                sec_masks = [erode_masks(sec_mask) for sec_mask in sec_masks]
                sec_masks = list(zip(sec_masks, sec_channels))
            else:
                sec_masks = None
            pair_arrays.append(((primary_mask, primary_name), sec_masks))
    return pair_arrays




def load_reference_masks(chunk_frames: range, ref_masks_fold: list[str] | None, exp_path: Path, pixel_resolution: float | None)-> list[tuple[np.ndarray, str, float | None]] | None:
    if ref_masks_fold is None:
        return None
    
    ref_masks = []
    for ref_fold in ref_masks_fold:
        ref_path = exp_path.joinpath(ref_fold)
        ref_files = sorted(ref_path.glob('*.tif'))
        ref_array = load_stack(ref_files, frame_range=chunk_frames, return_2D=True)
        ref_array = _dist_transform(ref_array)
        ref_name = ref_fold.split('_')[-1]
        ref_masks.append((ref_array, ref_name, pixel_resolution))
    return ref_masks

def load_diff_masks(img_array: np.ndarray)-> np.ndarray:
    # Extract the differencial data, as int16, to keep the negative values
    zero_array = np.zeros((1,) + img_array.shape[1:], dtype=np.int16)
    img_array_diff = np.concatenate((zero_array, np.diff(img_array.astype(np.int16), axis=0)), axis=0)
    return img_array_diff

def _dist_transform(mask: np.ndarray)-> np.ndarray:
    """Apply the distance transform to the mask."""
    
    
    return distance_transform_edt(np.logical_not(mask))    

def make_pairs(lst: list[T])-> list[tuple[T, list[T]]]:
    """Make pairs of elements from a list. For example, if the list is [1,2,3], the output will be [(1,[2,3]),(2,[1,3]),(3,[1,2])]."""
    
    pairs = []
    for i, element in enumerate(lst):
        others = lst[:i] + lst[i+1:]
        pairs.append((element, others))
    return pairs
