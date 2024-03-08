from __future__ import annotations
from os.path import join
from os import sep, PathLike
import shutil
from typing import Iterable
from tifffile import imread
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from pystackreg import StackReg
from image_handeling.Experiment_Classes import Experiment
from image_handeling.data_utility import load_stack, create_save_folder, save_tif
# TODO: replace imwrite with save_tif
# TODO: only apply chan_shift to the first frame
# TODO: save the transfo matrix to be able to apply it to the masks in the json file



def select_reg_mtd(reg_mtd: str)-> StackReg:
    mtd_list = ['translation','rigid_body','scaled_rotation','affine','bilinear']
    if reg_mtd not in mtd_list:
        raise ValueError(f"{reg_mtd} is not valid. Please only put {mtd_list}")
        
    if reg_mtd=='translation':       stackreg = StackReg(StackReg.TRANSLATION)
    elif reg_mtd=='rigid_body':      stackreg = StackReg(StackReg.RIGID_BODY)
    elif reg_mtd=='scaled_rotation': stackreg = StackReg(StackReg.SCALED_ROTATION)
    elif reg_mtd=='affine':          stackreg = StackReg(StackReg.AFFINE)
    elif reg_mtd=='bilinear':        stackreg = StackReg(StackReg.BILINEAR)
    return stackreg

def apply_tmat_to_img(input_dict: dict)-> None:
    """Transform the image with the given tmat and save it to the save_path."""
    img = imread(input_dict['img_path'])
    reg_img = input_dict['stackreg'].transform(img,input_dict['tmat'])
    reg_img[reg_img<0] = 0
    save_tif(reg_img.astype(np.uint16),input_dict['save_path'],**input_dict['metadata'])

def get_tmats_chan_dict(stackreg: StackReg, img_ref: np.ndarray, path_list: list[PathLike], 
                   channel_list: list[str], frame: int)-> dict[str,np.ndarray]:
    """Register the channel ref to every other channel imgs for the given frame. 
    Output is a dict with the channel as key and the tmat np.ndarray (2D) as value."""
    
    tmats_dict = {}
    for chan in channel_list:
        img = load_stack(path_list,chan,frame,True)
        tmats_dict[chan] = stackreg.register(img_ref,img)
    return tmats_dict

def get_tmats_frame_dict(stackreg: StackReg, img_ref: np.ndarray, path_list: list[PathLike],
                         frame_iterable: Iterable[int], reg_channel: str)-> dict[int,np.ndarray]:
    """Register the frame ref to every other frame imgs for the given channel.
    Output is a dict with the frame as key and the tmat np.ndarray (2D) as value."""
    tmats_dict = {}
    for frame in frame_iterable:
        img = load_stack(path_list,reg_channel,frame,True)
        tmats_dict[frame] = stackreg.register(img_ref,img)
    return tmats_dict

def sort_by_channel(file_list: list[PathLike], channel_list: list)-> dict[str,list[PathLike]]:
    """Sort the images by channel. Output is a dict with the channel as key and the list of images as value."""
    sorted_channel = {chan: [file for file in file_list if chan in file] for chan in channel_list}
    return sorted_channel

def sort_by_frame(file_list: list[PathLike], n_frames: int)-> dict[int,list[PathLike]]:
    """Sort the images by frame. Output is a dict with the frame as key and the list of images as value."""
    sorted_frames = {frame: [file for file in file_list if f"_f{frame+1:04d}" in file] for frame in range(n_frames)}
    return sorted_frames

def copy_first_frame(img_path: list[str], folder_name_dst: str)-> None:
    for path in img_path: 
        shutil.copyfile(path,path.replace('Images',folder_name_dst))


####################################################################################
############################# Channel shift correction #############################
####################################################################################

def correct_chan_shift(exp_set: Experiment, stackreg: StackReg, reg_channel: str)-> tuple[Experiment,dict[str,np.ndarray]]:
    # Load ref image
    img_ref = load_stack(exp_set.processed_images_list,reg_channel,0,True)
    
    # Sort the images by channel
    sorted_channel = sort_by_channel(exp_set.processed_images_list,exp_set.active_channel_list)
    del sorted_channel[reg_channel] # Remove the ref channel from the dict, as it doesn't need to be processed
    
    # Get all the tmats
    tmats_dict = get_tmats_chan_dict(stackreg,img_ref,exp_set.processed_images_list,sorted_channel.keys(),0)
    
    # Generate input data for parallel processing
    input_data = [{'stackreg':stackreg,
                   'img_path':path,
                   'save_path':path,
                   'tmat':tmats_dict[chan],
                   'metadata':{'um_per_pixel':exp_set.analysis.um_per_pixel,
                               'finterval':exp_set.analysis.interval_sec}} 
                  for chan, img_path_lst in sorted_channel.items() for path in img_path_lst]
    
    with ProcessPoolExecutor() as executor:
        executor.map(apply_tmat_to_img,input_data)
    return exp_set

############################# main function #############################
def register_channel_shift(exp_set_list: list[Experiment], reg_mtd: str, reg_channel: str, chan_shift_overwrite: bool=False)-> list[Experiment]:
    for exp_set in exp_set_list:
        # Check if the channel shift was already applied
        if exp_set.process.channel_reg and not chan_shift_overwrite:
            print(f" --> Channel shift was already applied on the images with {exp_set.process.channel_reg}")
            continue
        
        if len(exp_set.active_channel_list)==1:
            print(f" --> Only one channel in the active_channel_list, no need to apply channel shift")
            continue
        
        stackreg = select_reg_mtd(reg_mtd)
        print(f" --> Applying channel shift correction on the images with '{reg_channel}' as reference and {reg_mtd} methods")
        
        # If not, correct the channel shift
        exp_set = correct_chan_shift(exp_set,stackreg,reg_channel)
        
        # Save settings
        exp_set.process.channel_reg = [f"reg_channel={reg_channel}",f"reg_mtd={reg_mtd}"]
        # exp_set.process.ndarrayJSONencoder(tmats_dict)
        exp_set.save_as_json()
    return exp_set_list

####################################################################################
############################## Frame shift correction ##############################
####################################################################################
def register_with_first(stackreg: StackReg, exp_set: Experiment, reg_channel: str, method: str)-> None:
    # Load ref image
    if method=='first':
        img_ref = load_stack(exp_set.processed_images_list,reg_channel,0,True)
    elif method=='mean':
        img_ref = np.mean(load_stack(exp_set.processed_images_list,reg_channel,range(exp_set.img_properties.n_frames)),axis=0)
    
    # Get all transfo matrix for the ref channel, not need to process the first frame if method is 'first'
    frame_range = range(1,exp_set.img_properties.n_frames) if method=='first' else range(exp_set.img_properties.n_frames)
    tmats_dict = get_tmats_frame_dict(stackreg,img_ref,exp_set.processed_images_list,frame_range,reg_channel)
    
    # Generate input data for parallel processing
    sorted_frame = sort_by_frame(exp_set.processed_images_list,exp_set.img_properties.n_frames)
    if method=='first':
        copy_first_frame(sorted_frame[0],"Images_Registered") # Copy the first frame to the reg_folder
        del sorted_frame[0] # Remove the first frame from the dict, as it doesn't need to be processed
    
    input_data = [{'stackreg':stackreg,
                   'img_path':path,
                   'save_path':path.replace('Images','Images_Registered'),
                   'tmat':tmats_dict[frame],
                   'metadata':{'um_per_pixel':exp_set.analysis.um_per_pixel,
                               'finterval':exp_set.analysis.interval_sec}} 
                  for frame, img_path_lst in sorted_frame.items() for path in img_path_lst]
    
    # Apply the transfo matrix to the images
    with ProcessPoolExecutor() as executor:
        executor.map(apply_tmat_to_img,input_data)
    return exp_set

def register_with_mean(stackreg: StackReg, exp_set: Experiment, reg_channel: str, img_folder: PathLike)-> None:
    # Load ref image
    img_ref = np.mean(load_stack(exp_set.processed_images_list,reg_channel,range(exp_set.img_properties.n_frames)),axis=0)
    
    
    
    serie = int(exp_set.exp_path.split(sep)[-1].split('_')[-1][1:])
    for f in range(exp_set.img_properties.n_frames):
        # Load image to register
        img = load_stack(img_list=exp_set.processed_images_list,channel_list=[reg_channel],frame_range=[f])
        if exp_set.img_properties.n_slices>1: img = np.amax(img,axis=0)
        # Get the transfo matrix
        tmats = stackreg.register(ref=img_ref,mov=img)
        
        for chan in exp_set.active_channel_list:
            # Load image to transform
            img = load_stack(img_list=exp_set.processed_images_list,channel_list=[chan],frame_range=[f])
            for z in range(exp_set.img_properties.n_slices):
                # Apply transfo
                if exp_set.img_properties.n_slices==1: reg_img = stackreg.transform(mov=img,tmat=tmats)
                else: reg_img = stackreg.transform(mov=img[z,...],tmat=tmats)
                # Save
                reg_img[reg_img<0] = 0
                file_name = join(sep,img_folder+sep,chan+f"_s{serie:02d}"+'_f%04d'%(f+1)+'_z%04d.tif'%(z+1))
                save_tif(reg_img.astype(np.uint16),file_name,exp_set.analysis.um_per_pixel,exp_set.analysis.interval_sec)

def register_with_previous(stackreg: StackReg, exp_set: Experiment, reg_channel: str, img_folder: PathLike)-> None:
    for f in range(1,exp_set.img_properties.n_frames):
        # Load ref image
        if f==1:
            img_ref = load_stack(img_list=exp_set.processed_images_list,channel_list=[reg_channel],frame_range=[f-1])
            if exp_set.img_properties.n_slices>1: img_ref = np.amax(img_ref,axis=0)
        else:
            img_ref = load_stack(img_list=exp_set.register_images_list,channel_list=[reg_channel],frame_range=[f-1])
            if exp_set.img_properties.n_slices>1: img_ref = np.amax(img_ref,axis=0)
        
        # Load image to register
        img = load_stack(img_list=exp_set.processed_images_list,channel_list=[reg_channel],frame_range=[f])
        if exp_set.img_properties.n_slices>1: img = np.amax(img,axis=0)
        
        # Get the transfo matrix
        serie = int(exp_set.exp_path.split(sep)[-1].split('_')[-1][1:])
        tmats = stackreg.register(ref=img_ref,mov=img)
        for chan in exp_set.active_channel_list:
            # Load image to transform
            img = load_stack(img_list=exp_set.processed_images_list,channel_list=[chan],frame_range=[f])
            fst_img = load_stack(img_list=exp_set.processed_images_list,channel_list=[chan],frame_range=[f-1])
            for z in range(exp_set.img_properties.n_slices):
                # Copy the first image to the reg_folder
                if f==1:
                    if exp_set.img_properties.n_slices==1: 
                        first_file_name = join(sep,img_folder+sep,chan+f"_s{serie:02d}"+'_f%04d'%(1)+'_z%04d.tif'%(z+1))
                        save_tif(fst_img.astype(np.uint16),first_file_name,exp_set.analysis.um_per_pixel,exp_set.analysis.interval_sec)
                    else: 
                        first_file_name = join(sep,img_folder+sep,chan+f"_s{serie:02d}"+'_f%04d'%(1)+'_z%04d.tif'%(z+1))
                        save_tif(fst_img[z,...].astype(np.uint16),first_file_name,exp_set.analysis.um_per_pixel,exp_set.analysis.interval_sec)
                # Apply transfo
                if exp_set.img_properties.n_slices==1: reg_img = stackreg.transform(mov=img,tmat=tmats)
                else: reg_img = stackreg.transform(mov=img[z,...],tmat=tmats)
                # Save
                reg_img[reg_img<0] = 0
                file_name = join(sep,img_folder+sep,chan+f"_s{serie:02d}"+'_f%04d'%(f+1)+'_z%04d.tif'%(z+1)) 
                save_tif(reg_img.astype(np.uint16),file_name,exp_set.analysis.um_per_pixel,exp_set.analysis.interval_sec)

# # # # # # # # main functions # # # # # # # # # 
def register_img(exp_set_list: list[Experiment], reg_channel: str, reg_mtd: str, reg_ref: int, reg_overwrite: bool=False)-> list[Experiment]:
    for exp_set in exp_set_list:
        save_folder = create_save_folder(exp_set.exp_path,'Images_Registered')
        
        if exp_set.process.frame_reg and not reg_overwrite:
            print(f" --> Registration was already applied to the images with {exp_set.process.frame_reg}")
            continue
        
        stackreg = select_reg_mtd(reg_mtd)
        print(f" --> Registering images with '{reg_ref}_image' reference and {reg_mtd} method")
        if reg_ref=='first':
            register_with_first(stackreg,exp_set,reg_channel,save_folder)
        elif reg_ref=='previous':
            register_with_previous(stackreg,exp_set,reg_channel,save_folder)
        elif reg_ref=='mean':
            register_with_mean(stackreg,exp_set,reg_channel,save_folder)
        exp_set.process.frame_reg = [f"reg_channel={reg_channel}",f"reg_mtd={reg_mtd}",f"reg_ref={reg_ref}"]
        exp_set.save_as_json()
    return exp_set_list