from __future__ import annotations
import numpy as np
from skimage.morphology import disk, erosion
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from threading import Lock
from skimage.segmentation import expand_labels
from tqdm import trange


def erode_masks(mask: np.ndarray, pixel_rad: int = 6)-> np.ndarray:
    """Function to erode the secondary mask to minimize false positive overlap between primary mask cell and secondary cells. Mask will be eroded one cell at a time in parallel."""
    
    # Setup the erosion
    footprint = disk(pixel_rad)
    
    
    # Extract the number of frames
    nframes = mask.shape[0] if mask.ndim > 2 else 1
    
    # Erode the secondary mask
    for i in trange(nframes):
        unique_cells = np.unique(mask[i])[1:]
        with ThreadPoolExecutor() as executor:
            eroded_frames = executor.map(partial(_erode_mask,mask=mask[i],footprint=footprint,lock=Lock()),unique_cells)
        mask_frame = np.zeros_like(mask[i])
        for frame in eroded_frames:
            mask_frame += frame
        mask[i] = mask_frame
    return mask

def _erode_mask(cell_idx: int, mask: np.ndarray, footprint: np.ndarray, lock: Lock)-> np.ndarray:
    """Apply the erosion to the secondary mask for a single cell."""
    
    with lock:
        temp_mask = np.where(mask==cell_idx, cell_idx, 0)
    eroded_mask = erosion(temp_mask,footprint).astype('uint16')
    return eroded_mask

def dilate_masks(mask: np.ndarray, pixel_rad: int = 6)-> np.ndarray:
    """Function to dilate masks to make sure that the mb cell is included into the compartment mask."""
    
    
    # Dilate the mask
    with ThreadPoolExecutor() as executor:
        dilated_frames = executor.map(partial(expand_labels, distance=pixel_rad),mask)
    dilated_mask = np.zeros_like(mask)
    for i, frame in enumerate(dilated_frames):
        dilated_mask[i] = frame
    
    return dilated_mask

    
