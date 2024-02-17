from __future__ import annotations
from os.path import join, split
import numpy as np
import pandas as pd
from collections import Counter
from skimage.segmentation import expand_labels
from mask_transformation.mask_morph import morph_missing_mask
from Experiment_Classes import Experiment
from segmentation.cp_segmentation import parallel_executer
from loading_data import mask_list_src
from tifffile import imsave
from loading_data import load_stack, is_processed, create_save_folder, gen_input_data, delete_old_masks

def load_csv(channel_seg: str, csv_path: str, csv_name: str = None):
    
    if not csv_name:
        csv_name = channel_seg
    if not csv_name.endswith('.csv'):
        csv_name = csv_name + '.csv'
    
    print(f'Loading data from {csv_name}')
    
    csv_path = join(csv_path, csv_name)
        
    csv_data = pd.read_csv(csv_path, encoding= 'unicode_escape', sep=None, engine='python')
    
    return csv_data
        
def uniform_dataset(csv_data: pd.DataFrame, exp_set: Experiment): 
    #get values needed later in the run from the metadata
    interval = exp_set.analysis.interval_sec
    pixel_microns = exp_set.analysis.pixel_microns
    frames = exp_set.img_properties.n_frames
    
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
    
    if 'micron' in x_head:
        csv_data[x_head] = csv_data[x_head]/pixel_microns #recalculate from microns to pixel
    if 'micron' in y_head:    
        csv_data[y_head] = csv_data[y_head]/pixel_microns
        
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

def create_man_mask(img_dict: dict):
    #load image
    img = load_stack(img_dict['imgs_path'],img_dict['channel_seg_list'],[img_dict['frame']])
    
    # create emtpy mask based on images or mask size
    masks_man = np.zeros_like((img), dtype='uint16')
    
    #get small dataframe for all cell to be markted in this frame
    sorted = img_dict['csv_data'][img_dict['csv_data']['frame'] == img_dict['frame']]
    
    # mark the positions of the cells
    for _, row in sorted.iterrows():
        x = row['x']
        y = row['y']
        cell_number = row['TID'] 
        
        if masks_man[y, x] == 0: # check if other cell is on exact same position
            masks_man[y, x] = cell_number
        else: # move a bit to the side to apply cell position, try and except to stay inside the array, not to move out of bounce. 
              #There could potentially be another cell, but its higly unlikely cause we are talking about pixel.
            try: masks_man[y+1, x] = cell_number
            except: masks_man[y-1, x] = cell_number
            
    #dilate the cells to make them visible
    masks_man = expand_labels(masks_man, distance=img_dict['radius'])
    
    savedir = join(img_dict['exp_path'],'Masks_Manual_Track', split(img_dict['imgs_path'][0])[1])
    # Clean and save
    imsave(savedir,masks_man)

def get_freq(array, exclude): # function to find most frequent value in array, exclude value: for us 0
    count = Counter(array[array != exclude]).most_common(1)
    if not count:
        return exclude
    else:  
        return count[0][0]

def seg_track_manual(img_dict: dict):
    #load image
    mask = load_stack(img_dict['imgs_path'],img_dict['channel_seg_list'],[img_dict['frame']])
    
    # create emtpy mask based on images or mask size
    masks_man = np.zeros_like((mask), dtype='uint16')
    frame = img_dict['frame']
    tracked_mask = masks_man.copy()

    
    for cell_number in np.delete(np.unique(man_mask), np.where( np.unique(man_mask) == 0)):
        overlap_array = np.array([0]).astype('uint16')
        overlap_array = mask[man_mask==cell_number]
        if not any(np.unique(overlap_array)) != 0:
            temp_man_mask = man_mask.copy()
            loop = 0
        while not any(np.unique(overlap_array)) != 0: # dilate man_mask cell to get overlay with next possible cell. To correct unprecise manual tracking
            temp_man_mask = expand_labels(temp_man_mask, distance=1)
            overlap_array = mask[man_mask==cell_number]
            loop = loop+1
            if loop == img_dict['max_loop']:
                print(f'exiting after {loop} loops')
                break
        max_overlap = get_freq(overlap_array,0)
        tracked_mask[mask==max_overlap] = cell_number
        mask[mask==max_overlap] = 0
    
    savedir = join(img_dict['exp_path'],'Masks_Manual_Track', split(img_dict['imgs_path'][0])[1])
    # Clean and save
    imsave(savedir,tracked_mask)

def run_morph(exp_set:Experiment, mask_fold_src:str, channel_seg:str, n_mask:int):
    mask_src_list = mask_list_src(exp_set,mask_fold_src)
    mask_stack = load_stack(mask_src_list,[channel_seg],range(exp_set.img_properties.n_frames))
    mask_stack = morph_missing_mask(mask_stack, n_mask)
    
    # Save masks
    # mask_src_list = [file for file in mask_src_list if file.__contains__('_z0001')]
    for i,path in enumerate(mask_src_list):
        imsave(path,mask_stack[i,...].astype('uint16'))

# # # # # # # # main functions # # # # # # # # # 
def man_tracking(exp_set_list: list[Experiment], channel_seg: str, track_seg_mask: bool = False, mask_fold_src: str = None,
                csv_name: str = None, radius: int=5, morph: bool=True, n_mask=2, manual_track_overwrite: bool=False, max_loop: int = 10):
    
    for exp_set in exp_set_list:
        # Check if exist
        if is_processed(exp_set.masks.manual_tracking,channel_seg,manual_track_overwrite):
                # Log
            print(f" --> Cells have already been tracked for the '{channel_seg}' channel")
            continue
        
        # Track images
        print(f" --> Tracking cells for the '{channel_seg}' channel")
        
        # Create save folder and remove old masks
        create_save_folder(exp_set.exp_path,'Masks_Manual_Track')
        delete_old_masks(exp_set.masks.manual_tracking,channel_seg,exp_set.mask_manual_track_list,manual_track_overwrite)
        
        #load csv
        csv_data = load_csv(channel_seg=channel_seg, csv_name=csv_name, csv_path=exp_set.exp_path)
        
        #uniform the keys of the dataset and reduce the df to the necessary part and also do recalculations for time adn micron
        csv_data = uniform_dataset(csv_data=csv_data, exp_set=exp_set)
        
        #generate List of Data to run them through ParallelProcessing
        img_data = gen_input_data(exp_set, mask_fold_src, [channel_seg], radius=radius, max_loop=max_loop)


        parallel_executer(create_man_mask, img_data, False)
        if morph:
            run_morph(exp_set, mask_fold_src='Masks_Manual_Track', n_mask=n_mask)
        
        if track_seg_mask:          
            # do overwrite from seg mask
            parallel_executer(seg_track_manual, img_data, False)
            
        
        # Save settings
        exp_set.masks.manual_tracking[channel_seg] = {'mask_fold_src':mask_fold_src,'track_seg_mask':track_seg_mask,
                                        'csv_name':csv_name,'n_mask':n_mask,'morph':morph,'radius':radius}
        exp_set.save_as_json()
    return exp_set_list