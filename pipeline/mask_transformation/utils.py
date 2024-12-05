from __future__ import annotations
from pathlib import Path
import numpy as np
from skimage.morphology import disk, erosion
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from threading import Lock
from pipeline.utilities.data_utility import load_stack
from skimage.segmentation import expand_labels
from tifffile import imread
from tqdm import trange


def erode_masks(mask: np.ndarray, pixel_rad: int = 6)-> np.ndarray:
    """Function to erode the secondary mask to minimize false positive overlap between primary mask cell and secondary cells. Mask will be eroded one cell at a time in parallel."""
    
    # Setup the erosion
    footprint = disk(pixel_rad)
    
    # Erode the secondary mask
    unique_cells = np.unique(mask)[1:]
    with ThreadPoolExecutor() as executor:
        eroded_frame = executor.map(partial(_erode_mask,mask=mask,footprint=footprint,lock=Lock()),unique_cells)
    mask_frame = np.zeros_like(mask)
    for frame in eroded_frame:
        mask_frame += frame
    return mask_frame

def _erode_mask(cell_idx: int, mask: np.ndarray, footprint: np.ndarray, lock: Lock)-> np.ndarray:
    """Apply the erosion to the secondary mask for a single cell."""
    with lock:
        temp_mask = np.where(mask==cell_idx, cell_idx, 0)
    eroded_mask = erosion(temp_mask,footprint).astype('uint16')
    return eroded_mask

def dilate_masks(mask_path: Path, pixel_rad: int = 6)-> np.ndarray:
    """Function to dilate masks to make sure that the mb cell is included into the compartment mask."""
    
    # Load the mask
    mask = imread(mask_path)
    
    return expand_labels(mask, pixel_rad)

    
