from __future__ import annotations
from os import PathLike, sep
from os.path import join, split
import numpy as np
import pandas as pd
from collections import Counter
from skimage.segmentation import expand_labels
from pipeline.mask_transformation.complete_track import complete_track
from pipeline.utilities.Experiment_Classes import Experiment
from pipeline.utilities.data_utility import load_stack, is_processed, create_save_folder, delete_old_masks, seg_mask_lst_src, img_list_src, track_mask_lst_src
from tifffile import imsave
from concurrent.futures import ThreadPoolExecutor

def load_csv(channel_seg: str, csv_path: str, csv_name: str = None):
    
    if not csv_name:
        csv_name = channel_seg
    if not csv_name.endswith('.csv'):
        csv_name = csv_name + '.csv'
    
    print(f'Loading data from {csv_name}')
    
    csv_path = join(csv_path, csv_name)
    
    try:
        csv_data = pd.read_csv(csv_path, encoding= 'unicode_escape', sep=None, engine='python')
    except FileNotFoundError:
        raise FileNotFoundError(f'{csv_path} not found. Please check naming of the .csv file. Please either give a name of the csv in the input function or name the file according to the channel. I.E.: BF.csv, GFP.csv, ...')
        
    
    return csv_data
   
def gen_input_data_masks(exp_obj: Experiment, mask_fold_src: str, mask_fold_src2: str, channel_seg_list: list, **kwargs)-> list[dict]:
    # mask_fold_src, mask_list_src = seg_mask_lst_src(exp_obj,mask_fold_src) 
    # mask_path_list = mask_list_src(exp_obj,mask_fold_src)
    
    _, mask_path_list = seg_mask_lst_src(exp_obj,mask_fold_src) 
    mask_path_list2 = track_mask_lst_src(exp_obj,mask_fold_src2)
    
    # mask_fold_src, mask_list_src = seg_mask_lst_src(exp_obj,mask_fold_src2)     
    # mask_path_list2 = mask_list_src(exp_obj,mask_fold_src2)

    channel_seg = channel_seg_list[0]
    input_data = []
    for frame in range(exp_obj.img_properties.n_frames):
        input_dict = {}
        mask1_path = [mask for mask in mask_path_list if f"_f{frame+1:04d}" in mask and channel_seg in mask]
        input_dict['mask1_path'] = mask1_path
        mask2_path = [mask for mask in mask_path_list2 if f"_f{frame+1:04d}" in mask and channel_seg in mask]
        input_dict['mask2_path'] = mask2_path
        input_dict['frame'] = frame
        input_dict['channel_seg_list'] = channel_seg_list
        input_dict.update(kwargs)
        input_data.append(input_dict)
    return input_data   
        
def uniform_dataset(csv_data: pd.DataFrame, exp_obj: Experiment): 
    #get values needed later in the run from the metadata
    interval = exp_obj.analysis.interval_sec
    um_per_pixel = exp_obj.analysis.um_per_pixel
    frames = exp_obj.img_properties.n_frames
    
    # find the column keys for x, y and t
    x_head = csv_data.keys()[csv_data.keys().str.contains("x ")][0]
    y_head = csv_data.keys()[csv_data.keys().str.contains("y ")][0]
    
    # if there is only one frame, use track ID (TID) to iterate over the points
    if frames > 1:
        t_head = csv_data.keys()[csv_data.keys().str.contains("t ")][0]
        csv_data = csv_data[['TID',x_head,y_head,t_head]] # reduce dataframe to only needed columns        
    else:
        t_head = None
        csv_data = csv_data[['TID',x_head,y_head]] # reduce dataframe to only needed columns
    
    csv_data = csv_data.dropna() # drop random occuring empty rows from excel/csv
    csv_data = csv_data.astype(float)
    if 'micron' or '[µm]' in x_head:
        csv_data[x_head] = csv_data[x_head]/um_per_pixel[1] #recalculate from microns to pixel
    if 'micron' or '[µm]' in y_head:    
        csv_data[y_head] = csv_data[y_head]/um_per_pixel[0]
        
    # get framenumber out of timestamp (if frames > 1)
    if t_head:
        if 'min' in t_head:
            interval = interval/60
        elif 'h' in t_head:
            interval = interval/3600
        csv_data[t_head] = round(csv_data[t_head]/interval)
        
    # rename the keys to make them uniform, resulting dataframe as "TID", "x", "y", and "frame" if frames>1
    csv_data=csv_data.rename(columns={x_head:'x', y_head:'y', t_head: 'frame'})
    csv_data = csv_data.astype(int)
    
    return csv_data


def get_freq(array, exclude): # function to find most frequent value in array, exclude value: for us 0
    count = Counter(array[array != exclude]).most_common(1)
    if not count:
        return exclude
    else:  
        return count[0][0]

def seg_track_manual(img_dict: dict):
    #load masks
    mask_seg = load_stack(img_dict['mask1_path'],img_dict['channel_seg_list'],[img_dict['frame']])
    mask_track = load_stack(img_dict['mask2_path'],img_dict['channel_seg_list'],[img_dict['frame']])

    # create emtpy mask based on images or mask size
    tracked_mask = np.zeros_like((mask_seg), dtype='uint16')
    mask_track = expand_labels(mask_track, distance=img_dict['dilate_value'])
    values_in_frame = np.unique(mask_track)
    cells_in_frame = values_in_frame[values_in_frame!=0]
    for cell_number in cells_in_frame:
        overlap_array = mask_seg[mask_track==cell_number].flatten()
        max_overlap = get_freq(overlap_array,0)
        if not max_overlap == 0:
            tracked_mask[mask_seg==max_overlap] = cell_number
            mask_track[mask_track==cell_number] = 0
    # save new mask
    folder, filename = split(img_dict['mask1_path'][0])
    savedir = join(split(folder)[0],'Masks_Manual_Track', filename)
    imsave(savedir,tracked_mask)

def run_morph(exp_obj:Experiment, mask_fold_src:str, channel_seg:str, n_mask:int, copy_first_to_start: bool=True, copy_last_to_end: bool=True, ):
    mask_src_list = track_mask_lst_src(exp_obj,mask_fold_src)
    mask_stack = load_stack(mask_src_list,[channel_seg],range(exp_obj.img_properties.n_frames))
    mask_stack = complete_track(mask_stack, n_mask, copy_first_to_start, copy_last_to_end)
    
    # Save masks
    for i,path in enumerate(mask_src_list):
        imsave(path,mask_stack[i,...].astype('uint16'))

# # # # # # # # main functions # # # # # # # # # 
def man_tracking(exp_obj_lst: list[Experiment], channel_seg: str, track_seg_mask: bool = False, mask_fold_src: PathLike = None,
                csv_name: str = None, radius: int=5, copy_first_to_start: bool=True, copy_last_to_end: bool=True, mask_appear=2,
                dilate_value: int = 20, process_as_2D: bool=True,  overwrite: bool=False):
    """
    Perform Manual Tracking based on a csv file resulting of MTrackJ (ImageJ Plugin) on a list of experiments.

    Args:
        exp_obj_lst (list[Experiment]): List of Experiment objects to perform tracking on.
        channel_seg (str): Channel name for segmentation.
        track_seg_mask (bool): possibility to write tracks from csv onto automatically segmented masks. Defaults to False.
        mask_fold_src (PathLike, optional): Only nececarry if track_seg_mask==True. Source folder path for masks, where the track will be written on.
        csv_name (str, optional): name of the .csv file with the manualtracking information. If no name is given it takes the first .csv file in the folder.
        radius (int): Resulting cell radius of the created masks. Defaults to 5.
        copy_first_to_start (bool, optional): Flag to copy the first mask to the start. Defaults to True.
        copy_last_to_end (bool, optional): Flag to copy the last mask to the end. Defaults to True.
        mask_appear (int, optional): Number of times a mask should appear to be considered valid. Defaults to 2.
        dilate_value (int, optional): Only nececarry if track_seg_mask==True. Gives the area in which trackpoint is looking for a valide mask from the segmentation. Defaults to 20.
        overwrite (bool, optional): Flag to overwrite existing tracking results. Defaults to False.
        
    Returns:
        list[Experiment]: List of Experiment objects with updated tracking information.
    """
    if copy_last_to_end or copy_first_to_start:
        process_as_2D = True
    for exp_obj in exp_obj_lst:
        # Activate the branch
        exp_obj.tracking.is_manual_tracking = True
        # Already processed?
        if is_processed(exp_obj.tracking.manual_tracking,channel_seg,overwrite):
            print(f" --> Cells have already been tracked for the '{channel_seg}' channel")
            continue
        
        # Track images
        print(f" --> Tracking cells for the '{channel_seg}' channel")
        
        # Create save folder and remove old masks
        create_save_folder(exp_obj.exp_path,'Masks_Manual_Track')
        delete_old_masks(exp_obj.tracking.manual_tracking,channel_seg,exp_obj.man_tracked_masks_lst,overwrite)
        
        #load csv
        csv_data = load_csv(channel_seg=channel_seg, csv_name=csv_name, csv_path=exp_obj.exp_path)
        
        #uniform the keys of the dataset and reduce the df to the necessary part and also do recalculations for time and micron
        csv_data = uniform_dataset(csv_data=csv_data, exp_obj=exp_obj)
        
        #get path of image
        img_fold_src = None
        _, img_path_lst = img_list_src(exp_obj,img_fold_src)
        img = load_stack(img_path_lst,channel_seg,[0])
        #check if 3D
        
        if not exp_obj.img_properties.n_slices == 1:
            masks_man_o = np.zeros((img.shape[1], img.shape[2]), dtype='uint16')
        else:
            masks_man_o = np.zeros_like((img), dtype='uint16')
            process_as_2D = True
    
        def create_man_mask(frame:int):
            #load image
            masks_man = masks_man_o.copy()
            #create emtpy mask based on images or mask size
            #get small dataframe for all cell to be markted in this frame
            sorted = csv_data[csv_data['frame'] == frame]
            #mark the positions of the cells
            for _, row in sorted.iterrows():
                x = row['x']
                y = row['y']
                cell_number = row['TID'] 
                if masks_man[y, x] == 0: # check if other cell is on exact same position
                    masks_man[y, x] = cell_number
                else: #move a bit to the side to apply cell position, try and except to stay inside the array, not to move out of bounce. 
                    #There could potentially be another cell, but its higly unlikely cause we are talking about pixel.
                    try: masks_man[y+1, x] = cell_number
                    except: masks_man[y-1, x] = cell_number
            #dilate the cells to make them visible
            masks_man = expand_labels(masks_man, distance=dilate_value)
            series = int(exp_obj.exp_path.split('_')[-1][1:])
            if not process_as_2D:
                for z_slice in range(exp_obj.analysis.n_slices):
                    filename = channel_seg+'_s%02d'%(series)+'_f%04d'%(frame+1)+'_z%04d'%(z_slice+1)+'.tif'
                    savedir = join(exp_obj.exp_path,'Masks_Manual_Track', filename)
                    #save
                    imsave(savedir,masks_man) 
            else:
                filename = channel_seg+'_s%02d'%(series)+'_f%04d'%(frame+1)+'_z0001.tif'
                savedir = join(exp_obj.exp_path,'Masks_Manual_Track', filename)
                #save
                imsave(savedir,masks_man) 
        
        frame_list = range(exp_obj.img_properties.n_frames)
        with ThreadPoolExecutor() as executor:
            executor.map(create_man_mask,frame_list)
            
        if track_seg_mask:
            print('Applying manual tracks to the original mask')
            # do overwrite from seg mask
            img_data = gen_input_data_masks(exp_obj, mask_fold_src, mask_fold_src2='Masks_Manual_Track', channel_seg_list=[channel_seg], dilate_value=dilate_value)
            # seg_track_manual(img_data)
            with ThreadPoolExecutor() as executor:
                executor.map(seg_track_manual,img_data)

        run_morph(exp_obj, mask_fold_src='Masks_Manual_Track', channel_seg=channel_seg, n_mask=mask_appear, copy_first_to_start=copy_first_to_start, copy_last_to_end=copy_last_to_end)
        
            
        # Save settings
        exp_obj.tracking.manual_tracking[channel_seg] = {'mask_fold_src':mask_fold_src,'track_seg_mask':track_seg_mask,
                                        'csv_name':csv_name,'mask_appear':mask_appear, 'copy_first_to_start': copy_first_to_start, 'copy_last_to_end': copy_last_to_end,'radius':radius}
        exp_obj.save_as_json()
    return exp_obj_lst