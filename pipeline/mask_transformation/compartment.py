from __future__ import annotations
from pathlib import Path
from threading import Lock
from pipeline.mask_transformation.utils import erode_masks, dilate_masks
from pipeline.utilities.data_utility import run_multithread
from tifffile import imwrite, imread

################### Main function ###################
def compartment_mask(exp_path: Path, mask_folder: str, pixel_rad: int = 6, dilation_rad: int | None = None, overwrite: bool = False)-> None:
    """Function to generate compartment mask from the primary mask. The mask can be dilated first to make sure that the mb cell is included into the compartment mask. Then the mask is eroded and the primary mask is subtracted from the eroded mask. Returns both the eroded mask and the subtracted mask."""
    
    # Prepare the multithread function
    fixed_args = {'pixel_rad': pixel_rad, 'dilation_rad': dilation_rad, 'overwrite': overwrite}
    
    # Get the mask files
    folder_path = exp_path.joinpath(mask_folder)
    print(f" --> Creating Masks Compartment for \033[94m{folder_path}\033[0m")
    run_multithread(_create_compartment_mask, list(folder_path.iterdir()), fixed_args)


################### Helper functions ###################
def _create_compartment_mask(mask_path: Path, lock: Lock, pixel_rad: int = 6, dilation_rad: int | None = None, overwrite: bool = False)-> None:
    
    # Prepare the save directory
    exp_path = mask_path.parent.parent
    mask_save_dir = exp_path.joinpath("Masks_Compartment")
    mask_save_dir.mkdir(exist_ok=True)
    cyto_name = f"cyto{mask_path.name}"
    mb_name = f"mb{mask_path.name}"
    
    if mask_save_dir.joinpath(mb_name).is_file() and not overwrite:
        return
    # Apply dilation if required
    if dilation_rad is not None:
        mask = dilate_masks(mask_path, dilation_rad)
    else:
        mask = imread(mask_path)
    
    # Erode the mask
    eroded_mask = erode_masks(mask, pixel_rad)
    
    # Subtract the primary mask from the eroded mask
    ring_mask = mask - eroded_mask
    
    # Save the masks
    exp_path = mask_path.parent.parent
    mask_save_dir = exp_path.joinpath("Masks_Compartment")
    mask_save_dir.mkdir(exist_ok=True)
    cyto_name = f"cyto{mask_path.name}"
    mb_name = f"mb{mask_path.name}"
    with lock:
        imwrite(mask_save_dir.joinpath(mb_name), ring_mask.astype('uint16'))
        imwrite(mask_save_dir.joinpath(cyto_name), eroded_mask.astype('uint16'))
    
    

        
    
################### Testing ###################   
if __name__ == "__main__":
    from time import time
    
    start = time()
    exp_path = Path('/home/Test_images/nd2/Run4/c4z1t91v1_s1')
    mask_folder = 'Masks_IoU_Track'
    pixel_rad = 6
    dilation_rad = None
    
    compartment_mask(exp_path, mask_folder, pixel_rad, dilation_rad)
    print(f"Time: {time()-start} s")
    
        
    
    
    