from __future__ import annotations
from pipeline.mask_transformation.utils import erode_masks, dilate_masks
import numpy as np



def compartment_mask(mask: np.ndarray, pixel_rad: int = 6, dilation_rad: int | None = None)-> np.ndarray:
    """Function to generate compartment mask from the primary mask. The mask can be dilated first to make sure that the mb cell is included into the compartment mask. Then the mask is eroded and the primary mask is subtracted from the eroded mask. Returns both the eroded mask and the subtracted mask."""
    
    # Apply dilation if required
    if dilation_rad:
        mask_stack = dilate_masks(mask, dilation_rad)
    else:
        mask_stack = mask.copy()
    
    # Erode the mask
    eroded_mask = erode_masks(mask, pixel_rad)
    
    # Subtract the primary mask from the eroded mask
    ring_mask = mask_stack - eroded_mask
    
    return ring_mask, eroded_mask
    # return mask
    
    
    
    
    
if __name__ == "__main__":
    from tifffile import imread, imwrite
    from skimage.morphology import disk, erosion
    
    mask = imread("/home/Test_images/masks/similar_masks.tif")
    # mask[mask!=436] = 0
    
    # footprint = np.stack(disk(6)*mask.shape[0], axis=0)
    
    # eroded_mask = erosion(mask, footprint)
    ring, eroded_mask = compartment_mask(mask, dilation_rad=2)
    # eroded_mask = compartment_mask(mask, dilation_rad=6)
    imwrite("/home/Test_images/masks/similar_masks_ero.tif", eroded_mask)
    # eroded_mask = imread("/home/Test_images/masks/similar_masks_ero.tif")
    # ring = mask - eroded_mask
    imwrite("/home/Test_images/masks/similar_masks_ring.tif", ring)