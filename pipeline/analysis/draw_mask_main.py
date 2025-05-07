from __future__ import annotations
from pathlib import Path

from pipeline.mask_transformation.complete_track import complete_track
from pipeline.analysis.draw_gui import draw_polygons, polygon_into_mask
from pipeline.utilities.data_utility import create_save_folder, load_stack, save_tif


def draw_wound_mask(img_files: list[Path], mask_label: list[str] | str, channel_show: str, 
                    frames: int, overwrite: bool=False, **kwargs)-> None:
    """Function to draw a mask on the given Image. Will be saved in a folder.
    Args:
        exp_set (Experiment): The experiment settings.
        mask_label (str or list[str]):labels for the masks to be created.
        channel_show (str): channel that is shown for drawing the mask
        img_fold_src (str, optional): Images folder, from where the displayed image is loaded
        overwrite (bool): Flag to override.
    Returns:
        None, saves the masks into folder."""
    
    if isinstance(mask_label, str):
        mask_label = [mask_label]
    
    # Make sure that the img_files are Path objects
    img_files = [Path(file) for file in img_files]
    filtered_files = sorted(file for file in img_files if channel_show in file.name)
    
    # Check if mask_label exist
    exp_path = filtered_files[0].parent.parent
    for label in mask_label:
        label_path = Path(create_save_folder(exp_path,f'Masks_{label}'))
        if any(label_path.iterdir()) and not overwrite:
            print(f" --> Masks already exist for {label} in {exp_path}.")
            continue
        
        print(f" --> Drawing mask with label {mask_label}")
        # load image stack and transform it into an RGB format  
        img_stack = load_stack(filtered_files, frame_range=range(frames), return_2D=True)

        # Draw the polygons
        poly_dict = draw_polygons(img_stack)
        
        if not poly_dict:
            raise AttributeError('No mask drawn!')
        mask_stack = polygon_into_mask(poly_dict, img_stack.shape)
        mask_stack = complete_track(mask_stack, mask_appear=1, copy_first_to_start=True, copy_last_to_end=True)
        
        if kwargs and 'metadata' in kwargs:
            metadata = kwargs['metadata']
        else:
            metadata = {'finterval':None, 'um_per_pixel':None}
        
        # Save the masks
        filtered_files_z1 =sorted(file for file in filtered_files if '_z0001' in file.name)
        for frame, mask in enumerate(mask_stack):
            file_name = filtered_files_z1[frame].name
            save_path = label_path.joinpath(file_name)
            save_tif(mask, save_path, **metadata)