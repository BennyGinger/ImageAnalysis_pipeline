from __future__ import annotations
import shutil
from tifffile import imread
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from pystackreg import StackReg
from pipeline.image_handeling.Experiment_Classes import Experiment
from pipeline.image_handeling.data_utility import load_stack, create_save_folder, save_tif, is_processed


####################################################################################
############################# Channel shift correction #############################
################################## main function ###################################
def correct_channel_shift(exp_obj_lst: list[Experiment], reg_mtd: str, reg_channel: str, overwrite: bool=False)-> list[Experiment]:
    """Main function to apply the channel shift correction to the images."""
    for exp_obj in exp_obj_lst:
        # Activate the branch
        exp_obj.preprocess.is_channel_reg = True
        # Already processed?
        if is_processed(exp_obj.preprocess.channel_reg,overwrite=overwrite):
            print(f" --> Channel shift was already applied on the images with {exp_obj.preprocess.channel_reg}")
            continue
        # Or if it's needed
        if len(exp_obj.active_channel_list)==1:
            print(f" --> Only one channel in the active_channel_list, no need to apply channel shift")
            continue
        # If not, correct the channel shift
        stackreg = select_reg_mtd(reg_mtd)
        print(f" --> Applying channel shift correction on the images with '{reg_channel}' as reference and {reg_mtd} methods")
        # Apply the channel shift correction
        apply_chan_shift(exp_obj,stackreg,reg_channel)
        # Save settings
        exp_obj.preprocess.channel_reg = [f"reg_channel={reg_channel}",f"reg_mtd={reg_mtd}","fold_src=Images"]
        exp_obj.save_as_json()
    return exp_obj_lst

################################ Satelite functions ################################
def get_tmats_chan(stackreg: StackReg, exp_obj: Experiment, reg_channel: str)-> dict[str,np.ndarray]:
    """Register the first frame of all channels to the ref channel. 
    Output is a dict with the channel (excluding the ref channel) as key and the tmat np.ndarray (2D) as value."""
    # Load ref image
    img_ref = load_stack(exp_obj.ori_imgs_lst,reg_channel,0,True)
    # Get the list of channel to process
    channel_lst = [chan for chan in exp_obj.active_channel_list if chan!=reg_channel]
    # Get all the tmats
    tmats_dict = {}
    for chan in channel_lst:
        img = load_stack(exp_obj.ori_imgs_lst,chan,0,True)
        tmats_dict[chan] = stackreg.register(img_ref,img)
    return tmats_dict

def apply_chan_shift(exp_obj: Experiment, stackreg: StackReg, reg_channel: str)-> None:
    """Apply the channel shift correction to the images."""
    # Get all the tmats
    tmats_dict = get_tmats_chan(stackreg,exp_obj,reg_channel)
    
    # Sort the images by channel, expcept the reg_channel, as it doesn't need to be processed
    sorted_channels = {chan: [file for file in exp_obj.ori_imgs_lst if chan in file] 
                      for chan in exp_obj.active_channel_list if chan!=reg_channel} 
    
    # Generate input data for parallel processing
    input_data = [{'stackreg':stackreg,
                   'img_path':path,
                   'save_path':path,
                   'tmat':tmats_dict[chan],
                   'metadata':{'um_per_pixel':exp_obj.analysis.um_per_pixel,
                               'finterval':exp_obj.analysis.interval_sec}} 
                  for chan, img_path_lst in sorted_channels.items() for path in img_path_lst]
    
    with ProcessPoolExecutor() as executor:
        executor.map(apply_tmat_to_img,input_data)


####################################################################################
############################## Frame shift correction ##############################
################################## main function ###################################
def correct_frame_shift(exp_obj_lst: list[Experiment], reg_channel: str, reg_mtd: str, img_ref: str, overwrite: bool=False)-> list[Experiment]:
    """Main function to apply the frame shift correction to the images."""
    for exp_obj in exp_obj_lst:
        # Activate the branch
        exp_obj.preprocess.is_frame_reg = True
        create_save_folder(exp_obj.exp_path,'Images_Registered')
        # Check if the frame shift was already applied
        if is_processed(exp_obj.preprocess.frame_reg,overwrite=overwrite):
            print(f" --> Registration was already applied to the images with {exp_obj.preprocess.frame_reg}")
            continue
        # Or if it's needed
        if exp_obj.img_properties.n_frames==1:
            print(f" --> Only one frame in the image, no need to apply frame shift")
            continue
        # If not, correct the frame shift
        stackreg = select_reg_mtd(reg_mtd)
        print(f" --> Registering images with '{img_ref}_image' reference and {reg_mtd} method")
        # Apply the frame shift correction
        apply_frame_shift(stackreg,exp_obj,reg_channel,img_ref)
        # Save settings
        exp_obj.preprocess.frame_reg = [f"reg_channel={reg_channel}",f"reg_mtd={reg_mtd}",f"img_ref={img_ref}","fold_src=Images"]
        exp_obj.save_as_json()
    return exp_obj_lst

################################ Satelite functions ################################
def get_tmats_first(stackreg: StackReg, exp_obj: Experiment, reg_channel: str)-> dict[int,np.ndarray]:
    """Register all frames of the given channel compared to the first frame.
    Output is a dict with the frame (excluding the first one) as key and the tmat np.ndarray (2D) as value."""
    # Load ref image, that is the first frame
    img_ref = load_stack(exp_obj.ori_imgs_lst,reg_channel,0,True)
    # Get the frame range, don't need to process the first frame
    frame_range = range(1,exp_obj.img_properties.n_frames) 
    # Get all the tmats
    tmats_dict = {}
    for frame in frame_range:
        img = load_stack(exp_obj.ori_imgs_lst,reg_channel,frame,True)
        tmats_dict[frame] = stackreg.register(img_ref,img)
    return tmats_dict

def get_tmats_mean(stackreg: StackReg, exp_obj: Experiment, reg_channel: str)-> dict[int,np.ndarray]:
    """Register all frames of the given channel compared to the mean of all frames.
    Output is a dict with the frame as key and the tmat np.ndarray (2D) as value."""
    # Load ref image, that is the mean of the all frames
    img_ref = np.mean(load_stack(exp_obj.ori_imgs_lst,reg_channel,range(exp_obj.img_properties.n_frames),True),axis=0)
    # Get the frame range, that is all the frames
    frame_range = range(exp_obj.img_properties.n_frames)
    # Get all the tmats
    tmats_dict = {}
    for frame in frame_range:
        img = load_stack(exp_obj.ori_imgs_lst,reg_channel,frame,True)
        tmats_dict[frame] = stackreg.register(img_ref,img)
    return tmats_dict

def get_tmats_previous(stackreg: StackReg, exp_obj: Experiment, reg_channel: str)-> dict[int,np.ndarray]:
    """Register all frames of the given channel compared to the previous frame.
    Output is a dict with the frame (excluding the first one) as key and the tmat np.ndarray (2D) as value."""
    tmats_dict = {}
    for frame in range(1,exp_obj.img_properties.n_frames):
        # Load ref image
        if frame==1: # For the second frame, use the first frame as ref
            img_ref = load_stack(exp_obj.ori_imgs_lst,reg_channel,frame-1,True)
        else: # For the other frames, use the previous transformed image as ref
            img_ref = tr_img 
        # Load image to register
        img = load_stack(exp_obj.ori_imgs_lst,reg_channel,frame,True)
        # Register and transform img
        tr_img = stackreg.register_transform(img_ref,img)
        # Save the tmat
        tmats_dict[frame] = stackreg.get_matrix()
    return tmats_dict
    
def copy_first_frame(img_path: list[str], folder_name_dst: str)-> None:
    """Simple function to copy the first frame to the given folder name."""
    for path in img_path: 
        shutil.copyfile(path,path.replace('Images',folder_name_dst))

def apply_frame_shift(stackreg: StackReg, exp_obj: Experiment, reg_channel: str, img_ref: str)-> None:
    """Apply the frame shift correction to the images depending on the img_ref (first, mean or previous)."""
    # Load ref image
    if img_ref=='first':
        tmats_dict = get_tmats_first(stackreg,exp_obj,reg_channel)
    elif img_ref=='mean':
        tmats_dict = get_tmats_mean(stackreg,exp_obj,reg_channel)
    elif img_ref=='previous':
        tmats_dict = get_tmats_previous(stackreg,exp_obj,reg_channel)
    
    # Sort the images by frame
    sorted_frames = {frame: [file for file in exp_obj.ori_imgs_lst if f"_f{frame+1:04d}" in file] 
                    for frame in range(exp_obj.img_properties.n_frames)}
    # Copy the first frame to the reg_folder, then remove it from the dict, as it doesn't need to be processed
    if img_ref!='mean':
        copy_first_frame(sorted_frames[0],"Images_Registered") 
        del sorted_frames[0]
    
    input_data = [{'stackreg':stackreg,
                   'img_path':path,
                   'save_path':path.replace('Images','Images_Registered'),
                   'tmat':tmats_dict[frame],
                   'metadata':{'um_per_pixel':exp_obj.analysis.um_per_pixel,
                               'finterval':exp_obj.analysis.interval_sec}} 
                  for frame, img_path_lst in sorted_frames.items() for path in img_path_lst]
    
    # Apply the transfo matrix to the images
    with ProcessPoolExecutor() as executor:
        executor.map(apply_tmat_to_img,input_data)
    return exp_obj


####################################################################################
################################ Utility function ##################################
####################################################################################
def select_reg_mtd(reg_mtd: str)-> StackReg:
    """Select the registration method (Translation, Rigid Body, Scaled Rotation, Affine or Bilinear) 
    and return the stackreg object."""
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
