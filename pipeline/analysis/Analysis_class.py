from __future__ import annotations
from dataclasses import dataclass
from os import sep, remove
from os.path import join, exists
from pathlib import Path

import pandas as pd

from pipeline.analysis.extraction_module.pixel_wise import extract_pixelwise_data
from pipeline.mask_transformation.compartment import compartment_mask
from pipeline.utilities.Base_Module_Class import BaseModule
from pipeline.utilities.Experiment_Classes import Experiment
from pipeline.utilities.data_utility import img_list_src, seg_mask_lst_src, track_mask_lst_src
from pipeline.utilities.pipeline_utility import progress_bar, pbar_desc
from pipeline.analysis.data_extraction import extract_data
from pipeline.settings.Setting_Classes import Settings
from pipeline.analysis.draw_mask_main import draw_wound_mask

TRACKING_MASKS = ['iou_tracking','manual_tracking','gnn_tracking']

@dataclass
class AnalysisModule(BaseModule):
    ## Attributes from the BaseModule class:
        # input_folder: PathLike | list[PathLike]
        # exp_obj_lst: list[Experiment] = field(init=False)
        # optimization: bool = False
    
    def analyze_from_settings(self, settings: dict)-> pd.DataFrame:
        # If optimization is set, then process only the first experiment
        self.optimization = settings['optimization'] 
        
        # Analyze the data based on the settings
        sets = Settings(settings)
        if not hasattr(sets,'analysis'):
            print("No analysis settings found")
            return pd.DataFrame()
        
        sets = sets.analysis
        if hasattr(sets,'draw_mask'):
            self.draw_wound_mask(**sets.draw_mask)
        
        if hasattr(sets,'compartment_mask'):
            self.mask_compartment(**sets.compartment_mask)
            if hasattr(sets,'extract_data'):
                sets.extract_data['do_compart'] = True
        
        if hasattr(sets, 'extract_pixelwise'):
            self.extract_pixelwise_data(**sets.extract_pixelwise)
            self.save_as_json()
            return pd.DataFrame()
        
        if not hasattr(sets,'extract_data'):
            self.save_as_json()
            return pd.DataFrame()
        
        master_df = self.create_master_df(**sets.extract_data)
        self.save_as_json()
        return master_df
        
    def create_master_df(self, img_fold_src: str = "", mask_fold_src: list[str] | str = "", ref_mask_fold_src: list[str] | str = "", do_diff: bool=False, diff_channel_ratio: str | None=None, do_compart: bool = False, overwrite: bool=False)-> pd.DataFrame:
        all_dfs = []
        for exp_obj in progress_bar(self.exp_obj_lst_active,
                            desc=pbar_desc("Experiments"),
                            colour='blue'):
            # extract the data
            all_dfs.append(self.extract_data(exp_obj, img_fold_src, mask_fold_src, ref_mask_fold_src, do_diff, diff_channel_ratio, do_compart, overwrite))
        
        # Concatenate all the dataframes
        master_df = pd.concat(all_dfs)
        # Save the data
        save_path = join(self.input_folder,'master_df.csv')
        if exists(save_path):
            remove(save_path)
        master_df.to_csv(save_path,index=False)
        return master_df
    
    def extract_data(self, exp_obj: Experiment, img_fold_src: str = "", mask_fold_src: list[str] | str = "", ref_mask_fold_src: list[str] | str = "", do_diff: bool=False, diff_channel_ratio: str | None=None, do_compart: bool = False, overwrite: bool=False)-> pd.DataFrame:
        # Gather the images
        img_fold_src, img_paths = img_list_src(exp_obj, img_fold_src)
            
        # Gather the masks folders
        mask_fold_src = select_masks_fold_list(exp_obj, mask_fold_src)
        
        # Gather the reference masks folders, if any
        ref_mask_fold_src = load_ref_masks_list(exp_obj, ref_mask_fold_src)
        
        df = extract_data(img_paths=img_paths,
                          exp_path=Path(exp_obj.exp_path),
                          masks_fold=mask_fold_src,
                          do_diff=do_diff,
                          ref_masks_fold=ref_mask_fold_src,
                          pixel_resolution=exp_obj.analysis.um_per_pixel[0],
                          diff_channel_ratio=diff_channel_ratio,
                          do_compart=do_compart,
                          overwrite=overwrite)
        
        # Add the time in seconds and experiment name
        interval_sec = 1 if exp_obj.analysis.interval_sec is None else exp_obj.analysis.interval_sec
        df['time_sec'] = (df['frame']-1)*interval_sec
        exp_name = exp_obj.exp_path.rsplit(sep,1)[1]
        df['exp_name'] = exp_name
        # Add the labels
        for i,label in enumerate(exp_obj.analysis.labels):
            df[f'tag_level_{i}'] = label
        
        # Add the unique cell id
        merged_labels = "_".join([exp_name] + exp_obj.analysis.labels)
        df['cell_ID'] = merged_labels + "_" + df['cell_label'].astype(str)
        
        # Save the data
        exp_obj.analysis.analysis_type.update({'extract_data':{'img_fold_src':img_fold_src}})
        exp_obj.save_as_json()
        return df

    def draw_wound_mask(self, mask_label: str | list[str], channel_show: str=None, overwrite: bool=False)-> None:
        if isinstance(mask_label, str):
            mask_label=[mask_label]
        
        for exp_obj in self.exp_obj_lst_active:
            # Activate branch and get imgage files
            exp_obj.analysis.is_reference_masks = True
            img_flod_src, img_files = img_list_src(exp_obj,None)
            # Get metadata
            metadata = {'um_per_pixel':exp_obj.analysis.um_per_pixel,
                        'finterval':exp_obj.analysis.interval_sec}
            # Draw
            draw_wound_mask(img_files,mask_label,channel_show,exp_obj.img_properties.n_frames,overwrite,metadata=metadata)
            # Save settings
            exp_obj.analysis.reference_masks.update({label:{'fold_src':img_flod_src,'channel_show':channel_show} 
                                                     for label in mask_label})
            exp_obj.save_as_json()
    
    def mask_compartment(self, mask_fold_src: str = "", pixel_rad: int = 6, dilation_rad: int | None = None, overwrite: bool = False)-> None:
        # If optimization is set, then process only the first experiment
        exp_obj_lst = self.exp_obj_lst.copy()[:1] if self.optimization else self.exp_obj_lst
        
        
        for exp_obj in exp_obj_lst:
            # Activate branch
            exp_obj.analysis.is_compartment_masks = True
            
            # Load the masks folders, if not provided
            mask_fold_src = select_masks_fold_list(exp_obj, mask_fold_src)[0]
            
            # Create compartment masks
            compartment_mask(Path(exp_obj.exp_path), mask_fold_src, pixel_rad, dilation_rad, overwrite)
            
            # Save settings
            exp_obj.analysis.compartment_masks.update({'fold_src':mask_fold_src,'pixel_rad':pixel_rad,'dilation_rad':dilation_rad})
            exp_obj.save_as_json()

    def extract_pixelwise_data(self, mask_fold_src: str, mask_label: str, img_fold_src: str = "", mask_channel: str = "", overwrite: bool = False)-> None:
        """
        Extract pixelwise data from the given image paths and mask paths.
        Parameters:
            mask_fold_src (str): Folder name containing the masks that delimit the region of interest.
            mask_label (str): Label of the reference mask.
            img_fold_src (str): Folder name containing the images. If empty, will use the default folder.
            mask_channel (str): Mask channel to be used, if multiple channels exists. If not, leave it as "" (empty).
            overwrite (bool): If True, overwrite existing data.
        """
        
        for exp_obj in self.exp_obj_lst_active:
            exp_path = Path(exp_obj.exp_path)
            pix_resolution = exp_obj.analysis.um_per_pixel[0]
            interval_sec = exp_obj.analysis.interval_sec if exp_obj.analysis.interval_sec is not None else 1
            
            # Gather the images
            img_fold_src, img_paths = img_list_src(exp_obj, img_fold_src)
            img_paths = sorted(Path(file) for file in img_paths)
            
            # Gather the masks folders
            mask_fold = exp_path.joinpath(mask_fold_src)
            mask_files = sorted(mask_fold.glob("*.tif"))
            # Filter the mask files based on the mask channel. By default, mask_channel is empty so all files will be returned
            mask_paths = [mask_file for mask_file in mask_files if mask_channel in mask_file.name]
            
            # Gather the reference masks folders
            ref_mask_fold = exp_path.joinpath(f"Masks_{mask_label}")
            ref_paths = sorted(ref_mask_fold.glob("*.tif"))
            
            # Extract the pixelwise data
            extract_pixelwise_data(exp_path, img_paths, ref_paths, mask_paths, pix_resolution, interval_sec, overwrite)
            
            # Save the data
            exp_obj.analysis.analysis_type.update({'extract_pixelwise':{'img_fold_src':img_fold_src,
                                                                        'mask_fold_src':mask_fold_src,
                                                                        'mask_label':mask_label,
                                                                        'mask_channel':mask_channel}})
            exp_obj.save_as_json()

    
######################## Helper Functions ########################
def select_masks_fold_list(exp_obj: Experiment, mask_fold_src: list[str] | str)-> list[str]:
    if mask_fold_src:
        return mask_fold_src if isinstance(mask_fold_src, list) else [mask_fold_src]
    
    # Check if thracking masks are present
    if any(getattr(exp_obj.tracking, f"is_{key}") for key in TRACKING_MASKS):
        mask_fold_src = track_mask_lst_src(exp_obj, mask_fold_src)[0]
        return [mask_fold_src]
    
    # Else, retrun segmentation masks
    mask_fold_src = seg_mask_lst_src(exp_obj, mask_fold_src)[0]
    return [mask_fold_src]

def load_ref_masks_list(exp_obj: Experiment, ref_mask_fold_src: list[str] | str)-> list[str]:
    if ref_mask_fold_src:
        return ref_mask_fold_src if isinstance(ref_mask_fold_src, list) else [ref_mask_fold_src]
    
    # Check if reference masks are present
    if exp_obj.analysis.is_reference_masks:
        ref_mask_fold_src = exp_obj.analysis.reference_masks.keys()
        ref_mask_fold_src = [f"Masks_{mask}" for mask in ref_mask_fold_src]
        return ref_mask_fold_src
    
    return []




if __name__== "__main__":
    
    input_folder = "/home/Test_images/nd2/Run4"
    aclass = AnalysisModule(input_folder)
    
    aclass.create_master_df(img_fold_src="Images_Registered",
                            mask_fold_src=['Masks_IoU_Track'],
                            ref_mask_fold_src=None,
                            do_diff=True,
                            ratio_diff=None,
                            overwrite=True)