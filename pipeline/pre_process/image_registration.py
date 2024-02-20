from __future__ import annotations
from os.path import join
from os import sep, PathLike
from tifffile import imread, imwrite
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from pystackreg import StackReg
from image_handeling.Experiment_Classes import Experiment
from image_handeling.data_utility import load_stack, create_save_folder, save_tif
# TODO: replace imwrite with save_tif
# TODO: only apply chan_shift to the first frame
# TODO: save the transfo matrix to be able to apply it to the masks in the json file
def chan_shift_file_name(file_list: list[PathLike], channel_list: list, reg_channel: str)-> list[tuple]:
    """Return a list of tuples of file names to be registered. 
    The first element of the tuple is the reference image and the second is the image to be registered.
    """
    d = {chan: [file for file in file_list if chan in file] for chan in channel_list}
    chan_2b_process = channel_list.copy()
    chan_2b_process.remove(reg_channel)
    tuples_list = []
    for chan in chan_2b_process:
        ref_list = sorted(d[reg_channel])
        chan_list = sorted(d[chan])
        for i in range(len(ref_list)):
            tuples_list.append((ref_list[i],chan_list[i]))
    return tuples_list

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

def correct_chan_shift(input_dict: dict)-> None:
    # Load ref_img and img
    ref_img_path,img_path = input_dict['img_pairs']
    ref_img = imread(ref_img_path)
    img = imread(img_path)
    # Apply transfo
    reg_img = input_dict['stackreg'].register_transform(ref_img,img)
    # Save
    reg_img[reg_img<0] = 0
    imwrite(img_path,reg_img.astype(np.uint16))

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
                imwrite(join(sep,img_folder+sep,chan+f"_s{serie:02d}"+'_f%04d'%(f+1)+'_z%04d.tif'%(z+1)),reg_img.astype(np.uint16))

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
                imwrite(join(sep,img_folder+sep,chan+f"_s{serie:02d}"+'_f%04d'%(f+1)+'_z%04d.tif'%(z+1)),reg_img.astype(np.uint16))

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
                    if exp_set.img_properties.n_slices==1: imwrite(join(sep,img_folder+sep,chan+f"_s{serie:02d}"+'_f%04d'%(1)+'_z%04d.tif'%(z+1)),fst_img.astype(np.uint16))
                    else: imwrite(join(sep,img_folder+sep,chan+f"_s{serie:02d}"+'_f%04d'%(1)+'_z%04d.tif'%(z+1)),fst_img[z,...].astype(np.uint16))
                # Apply transfo
                if exp_set.img_properties.n_slices==1: reg_img = stackreg.transform(mov=img,tmat=tmats)
                else: reg_img = stackreg.transform(mov=img[z,...],tmat=tmats)
                # Save
                reg_img[reg_img<0] = 0
                imwrite(join(sep,img_folder+sep,chan+f"_s{serie:02d}"+'_f%04d'%(f+1)+'_z%04d.tif'%(z+1)),reg_img.astype(np.uint16))

# # # # # # # # main functions # # # # # # # # # 
def channel_shift_register(exp_set_list: list[Experiment], reg_mtd: str, reg_channel: str, chan_shift_overwrite: bool=False)-> list[Experiment]:
    for exp_set in exp_set_list:
        if exp_set.process.channel_shift_corrected and not chan_shift_overwrite:
            print(f" --> Channel shift was already applied on the images with {exp_set.process.channel_shift_corrected}")
            continue
        stackreg = select_reg_mtd(reg_mtd)
        print(f" --> Applying channel shift correction on the images with '{reg_channel}' as reference and {reg_mtd} methods")
        
        # Generate input data for parallel processing
        img_pairs_list = chan_shift_file_name(exp_set.processed_images_list,exp_set.active_channel_list,reg_channel)
        input_data = [{'stackreg':stackreg,'img_pairs':img_pairs} for img_pairs in img_pairs_list]
                
        with ProcessPoolExecutor() as executor:
            executor.map(correct_chan_shift,input_data)
        # Save settings
        exp_set.process.channel_shift_corrected = [f"reg_channel={reg_channel}",f"reg_mtd={reg_mtd}"]
        exp_set.save_as_json()
    return exp_set_list

def register_img(exp_set_list: list[Experiment], reg_channel: str, reg_mtd: str, reg_ref: int, reg_overwrite: bool=False)-> list[Experiment]:
    for exp_set in exp_set_list:
        save_folder = create_save_folder(exp_set.exp_path,'Images_Registered')
        
        if exp_set.process.img_registered and not reg_overwrite:
            print(f" --> Registration was already applied to the images with {exp_set.process.img_registered}")
            continue
        
        stackreg = select_reg_mtd(reg_mtd)
        print(f" --> Registering images with '{reg_ref}_image' reference and {reg_mtd} method")
        if reg_ref=='first':
            register_with_first(stackreg,exp_set,reg_channel,save_folder)
        elif reg_ref=='previous':
            register_with_previous(stackreg,exp_set,reg_channel,save_folder)
        elif reg_ref=='mean':
            register_with_mean(stackreg,exp_set,reg_channel,save_folder)
        exp_set.process.img_registered = [f"reg_channel={reg_channel}",f"reg_mtd={reg_mtd}",f"reg_ref={reg_ref}"]
        exp_set.save_as_json()
    return exp_set_list