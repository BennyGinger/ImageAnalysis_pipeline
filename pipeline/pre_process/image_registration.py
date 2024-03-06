from __future__ import annotations
from os.path import join
from os import sep, PathLike
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


####################################################################################
############################# Channel shift correction #############################
####################################################################################

def get_tmats_dict(sorted_img_dict: dict[str,list], stackreg: StackReg, channel_list: list, reg_channel: str)-> dict[str,np.ndarray]:
    """Register the ref channel img with every other channel imgs, for the first frame only. Output is a dict with
    the channel to be porcessed (excluding the ref channel) as key and the tmat np.ndarray (2D)."""
    
    # Get the channel to be processed
    chan_2b_process = channel_list.copy()
    chan_2b_process.remove(reg_channel)
    
    # Get the ref and mov images
    tmats_dict = {}
    for chan in chan_2b_process:
        ref_list = sorted(sorted_img_dict[reg_channel])
        chan_list = sorted(sorted_img_dict[chan])
        tmats_dict[chan] = stackreg.register(imread(ref_list[0]),imread(chan_list[0]))
    return tmats_dict

def sort_img_by_channel(file_list: list[PathLike], channel_list: list)-> dict[str,list[PathLike]]:
    """Sort the images by channel. Output is a dict with the channel as key and the list of images as value."""
    raw_imgs = {chan: [file for file in file_list if chan in file] for chan in channel_list}
    return raw_imgs

def apply_tmat_to_chan(input_dict: dict)-> None:
    img = imread(input_dict['img_path'])
    reg_img = input_dict['stackreg'].transform(img,input_dict['tmat'])
    reg_img[reg_img<0] = 0
    save_tif(reg_img.astype(np.uint16),input_dict['img_path'],*input_dict['metadata'])

def correct_chan_shift(exp_set: Experiment, stackreg: StackReg, reg_channel: str)-> tuple[Experiment,dict[str,np.ndarray]]:
    # Sort the images by channel
    sorted_img_dict = sort_img_by_channel(exp_set.processed_images_list,exp_set.active_channel_list)
    
    # Get the tmats for all channels
    tmats_dict = get_tmats_dict(sorted_img_dict,stackreg,exp_set.active_channel_list,reg_channel)
    
    # Generate input data for parallel processing
    del sorted_img_dict[reg_channel] # Remove the ref channel from the dict, as it doesn't need to be processed
    input_data = [{'stackreg':stackreg,
                   'img_path':path,
                   'tmat':tmats_dict[chan],
                   'metadata':(exp_set.analysis.um_per_pixel,exp_set.analysis.interval_sec)
                   } for chan, img_path_lst in sorted_img_dict.items() for path in img_path_lst]
    
    with ProcessPoolExecutor() as executor:
        executor.map(apply_tmat_to_chan,input_data)
    return exp_set,tmats_dict

############################# main function #############################
def register_channel_shift(exp_set_list: list[Experiment], reg_mtd: str, reg_channel: str, chan_shift_overwrite: bool=False)-> list[Experiment]:
    for exp_set in exp_set_list:
        # Check if the channel shift was already applied
        if exp_set.process.channel_reg and not chan_shift_overwrite:
            print(f" --> Channel shift was already applied on the images with {exp_set.process.channel_reg}")
            continue
        stackreg = select_reg_mtd(reg_mtd)
        print(f" --> Applying channel shift correction on the images with '{reg_channel}' as reference and {reg_mtd} methods")
        
        # If not, correct the channel shift
        exp_set, tmats_dict = correct_chan_shift(exp_set,stackreg,reg_channel)
        
        # Save settings
        exp_set.process.channel_reg = [f"reg_channel={reg_channel}",f"reg_mtd={reg_mtd}"]
        exp_set.process.ndarrayJSONencoder(tmats_dict)
        exp_set.save_as_json()
    return exp_set_list

####################################################################################
############################## Frame shift correction ##############################
####################################################################################

def register_with_first(stackreg: StackReg, exp_set: Experiment, reg_channel: str, img_folder: PathLike)-> None:
    # Load ref image
    img_ref = load_stack(img_list=exp_set.processed_images_list,channel_list=[reg_channel],frame_range=[0])
    if exp_set.img_properties.n_slices>1: img_ref = np.amax(img_ref,axis=0)
    
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

def register_with_mean(stackreg: StackReg, exp_set: Experiment, reg_channel: str, img_folder: PathLike)-> None:
    # Load ref image
    if exp_set.img_properties.n_slices==1: img_ref = np.mean(load_stack(img_list=exp_set.processed_images_list,channel_list=[reg_channel],frame_range=range(exp_set.img_properties.n_frames)),axis=0)
    else: img_ref = np.mean(np.amax(load_stack(img_list=exp_set.processed_images_list,channel_list=[reg_channel],frame_range=range(exp_set.img_properties.n_frames)),axis=1),axis=0)

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