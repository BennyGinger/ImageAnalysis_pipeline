from __future__ import annotations
from os import sep, listdir, PathLike
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
from os.path import join
from pipeline.image_handeling.Experiment_Classes import Experiment
from pipeline.image_handeling.data_utility import is_processed, seg_mask_lst_src, load_stack, create_save_folder, delete_old_masks
from pipeline.mask_transformation.complete_track import complete_track
from cellpose.utils import stitch3D
from cellpose.metrics import _intersection_over_union
from scipy.stats import mode
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tifffile import imwrite

def track_cells(masks: np.ndarray, stitch_threshold: float) -> np.ndarray:
    """
    Track cells in a sequence of masks. Using the Cellpose stitch3D function to stitch masks together 
    and then create a master mask to compare with each frame.

    Args:
        masks (np.ndarray): Array of masks representing cells in each frame.
        stitch_threshold (float): Threshold value for stitching masks together.

    Returns:
        np.ndarray: Master mask representing the stitched cells.

    """
    print('  ---> Tracking cells...')
    # basic stitching/tracking from Cellpose
    masks = stitch3D(masks, stitch_threshold)
    # Create mask with all the most common masks value
    master_mask = create_master_mask(masks)
    return master_mask_stitching(masks, master_mask, stitch_threshold)

def create_master_mask(masks: np.ndarray) -> np.ndarray:
    """
    Create a master mask by performing a 'mode' operation to get the most common value present in most t-frames per pixel. 
    Ignoring background by setting zero to nan. Therefore conversion to float is needed.

    Args:
        masks (np.ndarray): Input array of masks.

    Returns:
        np.ndarray: The master mask with all possible cells on one mask.
    """
    print('  ---> Creating master mask')
    rawmasks_ignorezero = masks.copy().astype(float)
    rawmasks_ignorezero[rawmasks_ignorezero == 0] = np.nan
    master_mask = mode(rawmasks_ignorezero, axis=0, keepdims=False, nan_policy='omit')[0]
    # Convert back to int
    return np.ma.getdata(master_mask).astype(int)

def master_mask_stitching(masks: np.ndarray, master_mask: np.ndarray, stitch_threshold: float) -> np.ndarray:
    """
    Second round of stitch/tracking by using a mastermask to compare with every frame.
    Stitch 2D masks into 3D volume with stitch_threshold on IOU.
    Slightly changed code from Cellpose 'stitch3D'.

    Args:
    masks (np.ndarray): Array of 2D masks.
    master_mask (np.ndarray): 2D master mask used for comparison.
    stitch_threshold (float): Threshold value for stitching.

    Returns:
    np.ndarray: Stitched masks as a 3D volume.
    """
    mmax = masks[0].max()
    empty = 0

    for i in range(len(masks)):
        iou = _intersection_over_union(masks[i], master_mask)[1:, 1:]
        if not iou.size and empty == 0:
            mmax = masks[i].max()
        elif not iou.size and not empty == 0:
            icount = masks[i].max()
            istitch = np.arange(mmax + 1, mmax + icount + 1, 1, int)
            mmax += icount
            istitch = np.append(np.array(0), istitch)
            masks[i] = istitch[masks[i]]
        else:
            iou[iou < stitch_threshold] = 0.0
            iou[iou < iou.max(axis=0)] = 0.0
            istitch = iou.argmax(axis=1) + 1
            ino = np.nonzero(iou.max(axis=1) == 0.0)[0]
            istitch[ino] = 0
            istitch = np.append(np.array(0), istitch)
            masks[i] = istitch[masks[i]]
            empty = 1
    return masks

def dice_coefficient(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Calculate the dice coefficient between two binary masks.

    Args:
    mask1 (np.ndarray): The first binary mask.
    mask2 (np.ndarray): The second binary mask.

    Returns:
    float: The dice coefficient between the two masks.
    """
    intersection = np.logical_and(mask1, mask2)
    return 2 * intersection.sum() / (mask1.sum() + mask2.sum())

def calculate_dice_coef(mask: np.ndarray, shape_thres_percent: float) -> np.ndarray:
    """
    Calculates the Dice coefficient for each mask in the input array and removes masks below the threshold.

    Args:
    mask (np.ndarray): Input array of masks.
    shape_thres_percent (float): Threshold percentage for mask similarity.

    Returns:
    np.ndarray: Array of masks after removing masks below the threshold.
    """
    # Convert to boolean
    mask_val = np.amax(mask)
    mask = mask.astype(bool)
    # Create a median mask as a ref
    med_mask = np.median(mask, axis=0)
    # Check mask similarity
    for i in range(mask.shape[0]):
        # Calculate dc
        dc = dice_coefficient(med_mask, mask[i])
        # Remove mask if below threshold
        if dc < shape_thres_percent:
            mask[i] = 0
    mask = mask.astype('uint16')
    mask[mask != 0] = mask_val
    return mask

def check_mask_similarity(mask: np.ndarray, shape_thres_percent: float = 0.9) -> np.ndarray:
    """
    Check the similarity of a mask by calculating the dice coefficient for each object in the mask.

    Args:
        mask (np.ndarray): The input mask with 3 dimensions.
        shape_thres_percent (float, optional): The threshold percentage for shape similarity. Defaults to 0.9.

    Returns:
        np.ndarray: The new mask with similarity scores over 0.9 for each object.

    Raises:
        TypeError: If the input mask does not contain 3 dimensions.
    """
    print('  ---> Checking mask similarity')
    # Check that mask contains 3 dimensions
    if mask.ndim != 3:
        raise TypeError(f"Input mask must contain 3 dimensions, {mask.ndim} dim were given")

    # Go through all mask_obj
    new_mask = np.zeros(shape=mask.shape)

    def process_mask(obj):
        # Isolate mask obj
        temp = mask.copy()
        temp[temp != obj] = 0
        # Calculate dice coef
        temp = calculate_dice_coef(temp, shape_thres_percent)
        return temp

    with ThreadPoolExecutor() as executor:
        obj_list = list(np.unique(mask))[1:]
        results = executor.map(process_mask, obj_list)

        for result in results:
            new_mask += result

    return new_mask
        
def reassign_mask_val(mask_stack: np.ndarray) -> np.ndarray:
    """
    Reassigns the values of the input mask stack to consecutive integers starting from 0.
    
    Args:
        mask_stack (np.ndarray): The input mask stack.
    
    Returns:
        np.ndarray: The mask stack with reassigned values.
    """
    print('  ---> Reassigning masks value')
    for n, val in enumerate(list(np.unique(mask_stack))):
        mask_stack[mask_stack == val] = n
    return mask_stack

def is_channel_in_lst(channel: str, img_paths: list[PathLike]) -> bool:
    """
    Check if a channel is in the list of image paths.

    Args:
        channel (str): The channel name.
        img_paths (list[PathLike]): The list of image paths.

    Returns:
        bool: True if the channel is in the list, False otherwise.
    """
    return any(channel in path for path in img_paths)

# # # # # # # # main functions # # # # # # # # # 
def iou_tracking(exp_obj_lst: list[Experiment], channel_seg: str, mask_fold_src: str,
                 stitch_thres_percent: float=0.5, shape_thres_percent: float=0.9,
                 overwrite: bool=False, mask_appear: int=5, copy_first_to_start: bool=True, 
                 copy_last_to_end: bool=True)-> list[Experiment]:
    """
    Perform IoU (Intersection over Union) based cell tracking on a list of experiments.

    Args:
        exp_obj_lst (list[Experiment]): List of Experiment objects to perform tracking on.
        channel_seg (str): Channel name for segmentation.
        mask_fold_src (str): Source folder path for masks.
        stitch_thres_percent (float, optional): Stitching threshold percentage. Defaults to 0.5. Higher values will result in more strict tracking (excluding more cells)
        shape_thres_percent (float, optional): Shape threshold percentage. Defaults to 0.9. Lower values will result in tracks with more differences in shape between frames.
        overwrite (bool, optional): Flag to overwrite existing tracking results. Defaults to False.
        mask_appear (int, optional): Number of times a mask should appear to be considered valid. Defaults to 5.
        copy_first_to_start (bool, optional): Flag to copy the first mask to the start. Defaults to True.
        copy_last_to_end (bool, optional): Flag to copy the last mask to the end. Defaults to True.
    
    Returns:
        list[Experiment]: List of Experiment objects with updated tracking information.
    """
    
    for exp_obj in exp_obj_lst:
        # Check if time sequence
        if exp_obj.img_properties.n_frames == 1:
            print(f" --> '{exp_obj.exp_path}' is not a time sequence")
            continue
        
        # Activate the branch
        exp_obj.tracking.is_iou_tracking = True
        # Already processed?
        if is_processed(exp_obj.tracking.iou_tracking,channel_seg,overwrite):
            print(f" --> Cells have already been tracked for the '{channel_seg}' channel")
            continue
        # Track images
        print(f" --> Tracking cells for the '{channel_seg}' channel")
        
        # Create save folder and remove old masks
        create_save_folder(exp_obj.exp_path,'Masks_IoU_Track')
        delete_old_masks(exp_obj.tracking.iou_tracking,channel_seg,exp_obj.iou_tracked_masks_lst,overwrite)
        
        # Load masks
        mask_fold_src, mask_src_list = seg_mask_lst_src(exp_obj,mask_fold_src)
        if not is_channel_in_lst(channel_seg,mask_src_list):
            print(f" --> Channel '{channel_seg}' not found in the mask folder")
            continue
        mask_stack = load_stack(mask_src_list,[channel_seg],range(exp_obj.img_properties.n_frames),True)
        
        # Track masks
        mask_stack = track_cells(mask_stack,stitch_thres_percent)
        # Check shape similarity to avoid false masks
        mask_stack = check_mask_similarity(mask_stack,shape_thres_percent)
        
        # Re-assign the new value to the masks and obj. Previous step may have created dicontinuous masks
        mask_stack = reassign_mask_val(mask_stack)
        
        # Morph missing masks
        mask_stack = complete_track(mask_stack,mask_appear,copy_first_to_start,copy_last_to_end)
        
        # Save masks
        mask_src_list = [file for file in mask_src_list if file.__contains__('_z0001')]
        for i,path in enumerate(mask_src_list):
            mask_path = path.replace('Masks','Masks_IoU_Track').replace('_Cellpose','').replace('_Threshold','')
            imwrite(mask_path,mask_stack[i,...].astype('uint16'))
        
        # Save settings
        exp_obj.tracking.iou_tracking[channel_seg] = {'fold_src':mask_fold_src,'stitch_thres_percent':stitch_thres_percent,
                                        'shape_thres_percent':shape_thres_percent,'n_mask':mask_appear}
        exp_obj.save_as_json()
    return exp_obj_lst



if __name__ == "__main__":
    folder = '/Users/benhome/BioTool/GitHub/cp_dev/Test_images/Run4/c4z1t91v1_s1/Masks_Cellpose'
    mask_folder_src = [join(sep,folder+sep,file) for file in sorted(listdir(folder)) if file.endswith('.tif')]
    mask_stack = load_stack(mask_folder_src,['RFP'],range(91))
    print(type(mask_stack))
