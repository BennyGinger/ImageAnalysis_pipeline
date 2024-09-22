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
    from pathlib import Path
    from tqdm import tqdm
    from pipeline.utilities.data_utility import load_stack
    
    # Get all the folders
    data_dir = Path("/home/Barna/Hyper7/H2O2_gradient")
    exp_folders = [file.parent for file in data_dir.rglob("exp_settings.json")]
    
    for exp in tqdm(exp_folders):
        # Load the mask
        mask_folder = exp.joinpath("Masks_IoU_Track")
        mask_files = sorted(list(mask_folder.glob("GFP*")))
        mask_stack = load_stack(mask_files, return_2D=True)
        
        # Generate the compartment mask
        ring_mask, eroded_mask = compartment_mask(mask_stack, pixel_rad=5, dilation_rad=2)
        
        # Save the masks
        mask_save_dir = exp.joinpath("Masks_Compartment")
        mask_save_dir.mkdir(exist_ok=True)
        for i in range(len(mask_files)):
            cyto_name = f"cyto{mask_files[i].name}"
            mb_name = f"mb{mask_files[i].name}"
            imwrite(mask_save_dir.joinpath(mb_name), ring_mask[i].astype('uint16'))
            imwrite(mask_save_dir.joinpath(cyto_name), eroded_mask[i].astype('uint16'))
    
    # fold = Path("/home/Barna/Hyper7/DUOX/240830_HyPer7_duox/ATP/cytoplasm/1840_634_100uM_ATP@2min_1uM_H2O2@5min004_s1")
    # mask_fold = fold.joinpath("Masks_IoU_Track")
    # mask_files = sorted(list(mask_fold.glob("GFP*")))
    
    # mask_stack = load_stack(mask_files, return_2D=True)
    
    # # Generate the compartment mask
    # ring_mask, eroded_mask = compartment_mask(mask_stack, pixel_rad=5, dilation_rad=2)
    
    # # Save the masks
    # mask_save_dir = fold.joinpath("Masks_Compartment")
    # mask_save_dir.mkdir(exist_ok=True)
    # for i in range(len(mask_files)):
    #     cyto_name = f"cyto{mask_files[i].name}"
    #     mb_name = f"mb{mask_files[i].name}"
    #     imwrite(mask_save_dir.joinpath(mb_name), ring_mask[i].astype('uint16'))
    #     imwrite(mask_save_dir.joinpath(cyto_name), eroded_mask[i].astype('uint16'))
        
    
    
    