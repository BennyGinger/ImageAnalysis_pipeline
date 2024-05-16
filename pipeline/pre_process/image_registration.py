from __future__ import annotations
from functools import partial
from os import PathLike, scandir, sep, listdir
import shutil
from tifffile import imread
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pystackreg import StackReg
from pipeline.image_handeling.Experiment_Classes import Experiment
from pipeline.image_handeling.data_utility import load_stack, create_save_folder, save_tif, is_processed


####################################################################################
############################# Channel shift correction #############################
################################## main function ###################################
def correct_channel_shift(img_paths: list[PathLike], reg_mtd: str, reg_channel: str, 
                          channels: list[str]=[], metadata: dict={})-> None:
    """Main function to apply the channel shift correction to the images. Requires multiple channels.
    It will register only the first frames of all channels to the reg_channel. If images have a z-stack, 
    it will only register the maxIP of the images, but the transformation will be applied to all images.
    
    Args:
        img_paths (list[PathLike]): The list of images path.
        
        reg_mtd (str): The registration method to use (Translation, Rigid Body, Scaled Rotation, Affine or Bilinear).
        
        reg_channel (str): The reference channel to register the other channels to.
        
        channels (list[str], optional): The list of channels to process. If empty, it will get all the channels from the img_paths. Defaults to [].
        
        metadata (dict, optional): The metadata (mainly the interval of the frames and resolution) to save with the images. Defaults to {}."""
    
    
    # Check channels list, if empty, get all the channels from the img_paths
    if not channels: 
        channels = list(set([get_channel_from_path(path) for path in img_paths]))
    
    # Check if only one channel is detected
    if len(channels)==1:
        print(f" --> Only one channel detected in the img_paths, no need to apply channel shift")
        return
    
    # Initiate the stackreg object
    stackreg = select_reg_mtd(reg_mtd)
    print(f"--> Applying channel shift correction on the images with '{reg_channel}' as reference and {reg_mtd} methods")
    
    # Apply the channel shift correction
    apply_chan_shift(stackreg,img_paths,channels,reg_channel,metadata)
    

################################ Satelite functions ################################
def get_tmats_chan(stackreg: StackReg, img_paths: list[PathLike], channels: list[str], reg_channel: str)-> dict[str,np.ndarray]:
    """Register the first frame of all channels to the ref channel. 
    Output is a dict with the channel (excluding the ref channel) as key and the tmat np.ndarray (2D) as value."""
    # Load ref image
    img_ref = load_stack(img_paths,reg_channel,0,True)
    # Get the list of channel to process
    channels.remove(reg_channel)
    # Get all the tmats
    tmats_dict = {}
    for chan in channels:
        img = load_stack(img_paths,chan,0,True)
        tmats_dict[chan] = stackreg.register(img_ref,img)
    return tmats_dict

def apply_chan_shift(stackreg: StackReg, img_paths: list[PathLike], channels: list[str], 
                     reg_channel: str, metadata: dict)-> None:
    """Apply the channel shift correction to the images."""
    # Get all the tmats
    tmats_dict = get_tmats_chan(stackreg,img_paths,channels,reg_channel)
    
    # Sort the images by channel, expcept the reg_channel, as it doesn't need to be processed
    img_paths = [path for path in img_paths if reg_channel not in path]
    
    # Generate input data for parallel processing
    if not metadata:
        metadata = {'um_per_pixel':None,'finterval':None}
    
    partial_apply_tmat = partial(apply_tmat_to_img,stackreg=stackreg,tmats_dict=tmats_dict,
                                 metadata=metadata,save_fold='Images')
    
    with ProcessPoolExecutor() as executor:
        executor.map(partial_apply_tmat,img_paths)


####################################################################################
############################## Frame shift correction ##############################
################################## main function ###################################
def correct_frame_shift(img_paths: list[PathLike], reg_channel: str, reg_mtd: str, img_ref: str, overwrite: bool=False,
                        metadata: dict={}, frames: int=None)-> list[Experiment]:
    """Main function to apply the frame shift correction to the images. Requires multiple frames.
    It will register all the frames to the first frame, the mean of all frames or the previous frame.
    If images have a z-stack, it will only register the maxIP of the images, but the transformation will 
    be applied to all images.
    
    Args:
        img_paths (list[PathLike]): The list of images path.
        
        reg_channel (str): The reference channel to register the other channels to.
        
        reg_mtd (str): The registration method to use (Translation, Rigid Body, Scaled Rotation, Affine or Bilinear).
        
        img_ref (str): The reference image to register the other images to (first, mean or previous).
        
        overwrite (bool, optional): If True, it will overwrite the images in the 'Images_Registered' folder. Defaults to False.
        
        metadata (dict, optional): The metadata (mainly the interval of the frames and resolution) to save with the images. Defaults to {}.
        
        frames (int, optional): The number of frames in the images. If None, it will get the number of frames from the img_paths. Defaults to None."""
    
    # Check frames number
    if not frames:
        frames = len(set([get_frame_from_path(path) for path in img_paths]))
    
    # Check if only one frame is detected
    if frames==1:
        print(f" ---> Only one frame detected in the img_paths, no need to apply frame shift")
        return
    
    # Set up the saving folder
    exp_path: PathLike = img_paths[0].rsplit(sep,2)[0]
    save_path = create_save_folder(exp_path,'Images_Registered')
    
    # Check if the frame shift was already applied
    if any(scandir(save_path)) and not overwrite:
        print("---> Registration was already applied to the images")
        return
    
    # Initiate the stackreg object
    stackreg = select_reg_mtd(reg_mtd)
    print(f" --> Registering images with '{img_ref}_image' reference and {reg_mtd} method")
    
    # Apply the frame shift correction
    apply_frame_shift(img_paths,stackreg,frames,reg_channel,img_ref,metadata)
    

################################ Satelite functions ################################
def register_img(frame: int, img_paths: list[PathLike], reg_channel: str, stackreg: StackReg, img_ref: np.ndarray)-> dict[int,np.ndarray]:
    img = load_stack(img_paths,reg_channel,frame,True)
    return {frame: stackreg.register(img_ref,img)}

def get_tmats_first(img_paths: list[str], frames: int, stackreg: StackReg, reg_channel: str)-> dict[int,np.ndarray]:
    """Register all frames of the given channel compared to the first frame.
    Output is a dict with the frame (excluding the first one) as key and the tmat np.ndarray (2D) as value."""
    # Load ref image, that is the first frame
    img_ref = load_stack(img_paths,reg_channel,0,True)
    # Get the frame range, don't need to process the first frame
    frame_range = range(1,frames) 
    # Get all the tmats
    partial_register_img = partial(register_img,reg_channel=reg_channel,stackreg=stackreg,img_ref=img_ref, img_paths=img_paths)
    with ThreadPoolExecutor() as executor:
        tmats = executor.map(partial_register_img,frame_range)
    tmat_dict = {k:v for tmat in tmats for k,v in tmat.items()}
    return tmat_dict

def get_tmats_mean(img_paths: list[str], frames: int, stackreg: StackReg, reg_channel: str)-> dict[int,np.ndarray]:
    """Register all frames of the given channel compared to the mean of all frames.
    Output is a dict with the frame as key and the tmat np.ndarray (2D) as value."""
    # Get the frame range, that is all the frames
    frame_range = range(frames)
    # Load ref image, that is the mean of the all frames
    img_ref = np.mean(load_stack(img_paths,reg_channel,frame_range,True),axis=0)
    # Get all the tmats
    partial_register_img = partial(register_img,reg_channel=reg_channel,stackreg=stackreg,img_ref=img_ref,img_paths=img_paths)
    with ThreadPoolExecutor() as executor:
        tmats_dict = executor.map(partial_register_img,frame_range)
    return {k:v for tmat_dict in tmats_dict for k,v in tmat_dict.items()}

def get_tmats_previous(img_paths: list[str], frames: int, stackreg: StackReg, reg_channel: str)-> dict[int,np.ndarray]:
    """Register all frames of the given channel compared to the previous frame.
    Output is a dict with the frame (excluding the first one) as key and the tmat np.ndarray (2D) as value."""
    tmats_dict = {}
    for frame in range(1,frames):
        # Load ref image
        if frame==1: # For the second frame, use the first frame as ref
            img_ref = load_stack(img_paths,reg_channel,frame-1,True)
        else: # For the other frames, use the previous transformed image as ref
            img_ref = tr_img 
        # Load image to register
        img = load_stack(img_paths,reg_channel,frame,True)
        # Register and transform img
        tr_img = stackreg.register_transform(img_ref,img)
        # Save the tmat
        tmats_dict[frame] = stackreg.get_matrix()
    return tmats_dict
    
def copy_first_frame(img_path: list[str], folder_name_dst: str)-> None:
    """Simple function to copy the first frame to the given folder name."""
    for path in img_path: 
        shutil.copyfile(path,path.replace('Images',folder_name_dst))

def apply_frame_shift(img_paths: list[PathLike], stackreg: StackReg, frames: int,
                      reg_channel: str, img_ref: str, metadata: dict)-> None:
    """Apply the frame shift correction to the images depending on the img_ref (first, mean or previous)."""
    # Load ref image
    if img_ref=='first':
        tmats_dict = get_tmats_first(img_paths,frames,stackreg,reg_channel)
    elif img_ref=='mean':
        tmats_dict = get_tmats_mean(img_paths,frames,stackreg,reg_channel)
    elif img_ref=='previous':
        tmats_dict = get_tmats_previous(img_paths,frames,stackreg,reg_channel)
    
    # Copy the first frame to the reg_folder, then remove it from the dict, as it doesn't need to be processed
    if img_ref!='mean':
        first_frames = [path for path in img_paths if '_f0001' in path]
        copy_first_frame(first_frames,"Images_Registered") 
        # Remove the first frame
        img_paths = [path for path in img_paths if '_f0001' not in path]
    
    if not metadata:
        metadata = {'um_per_pixel':None,'finterval':None}
    partial_apply_tmat = partial(apply_tmat_to_img,stackreg=stackreg,tmats_dict=tmats_dict,
                                 metadata=metadata, save_fold='Images_Registered')
    
    # Apply the transfo matrix to the images
    with ProcessPoolExecutor() as executor:
        executor.map(partial_apply_tmat,img_paths)

def get_channel_from_path(img_path: PathLike)-> str:
    """Get the channel name from the image path with the format 
    'path/to/image/channel_s00_f0000_z0000.tif'."""
    return img_path.rsplit(sep,1)[-1].split('_',1)[0]

def get_frame_from_path(img_path: PathLike)-> int:
    """Get the frame number from the image path with the format
    'path/to/image/channel_s00_f0000_z0000.tif'"""
    return int(img_path.rsplit(sep,1)[-1].split('_')[2:-1][0][1:])-1
     

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

def apply_tmat_to_img(img_path: PathLike, stackreg: StackReg, tmats_dict: dict[str,np.ndarray],
                      metadata: dict, save_fold: str)-> None:
    """Transform the image with the given tmat and save it to the save_path."""
    # Set up transformation
    img = imread(img_path)
    if isinstance(list(tmats_dict.keys())[0],str):
        key = get_channel_from_path(img_path)
    elif isinstance(list(tmats_dict.keys())[0],int):
        key = get_frame_from_path(img_path)
    
    # Apply transfomation
    reg_img = stackreg.transform(img,tmats_dict[key])
    reg_img[reg_img<0] = 0
    
    # If save_path is provided, use it, else save in place
    save_tif(reg_img.astype(np.uint16),img_path.replace('Images',save_fold),**metadata)

    

if __name__ == "__main__":
    from time import time
    from os.path import join
    # Test
    
    folder = '/home/Test_images/tiff/Run2/c2z25t23v1_tif_s1/Images'
    img_paths = [join(folder,file) for file in sorted(listdir(folder))]
    start = time()
    # correct_channel_shift(img_paths,'rigid_body','RFP')
    end = time()
    # print(f"Time taken: {end-start}")
    start = time()
    correct_frame_shift(img_paths,'RFP','rigid_body','previous',True)
    end = time()
    print(f"Time taken: {end-start}")