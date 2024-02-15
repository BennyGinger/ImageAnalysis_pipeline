from os.path import join,getsize
from os.path import isdir,exists
from os import mkdir,sep,scandir,listdir,remove
from nd2reader import ND2Reader
from tifffile import imread,imwrite,TiffFile
from skimage.morphology import disk,remove_small_objects,remove_small_holes
from mahotas import distance
import numpy as np
from smo import SMO
import pandas as pd
from pystackreg import StackReg
import cv2,pickle,itertools
from skimage import draw
from psutil import virtual_memory
from math import ceil
import mahotas as mh
from numbers import Number
from platform import system
from cellpose.metrics import _intersection_over_union
from cellpose.utils import stitch3D
from scipy.stats import mode,iqr
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sb
import json

def get_tif_meta(in_var): # [ ]: get tif metadata
    # Open tif and read meta
    with TiffFile(in_var) as tif:
        i_m = tif.imagej_metadata
        i_m['axes'] = tif.series[0].axes
        for page in tif.pages:
            for tag in page.tags:
                if tag.name in ['ImageWidth','ImageLength',]:
                    i_m[tag.name] = tag.value
                if tag.name in ['XResolution','YResolution']:
                    i_m[tag.name] = tag.value[0]/tag.value[1]

    # Calculate pixelsize
    xres,yres = i_m['XResolution'],i_m['YResolution']
    w_pixel,h_pixel = i_m['ImageWidth'],i_m['ImageLength']
    w_micron,h_micron = round(w_pixel/xres,ndigits=3),round(h_pixel/yres,ndigits=3)
    x_vox,y_vox = w_micron/w_pixel,h_micron/h_pixel
    if x_vox==y_vox:
        pixsize = x_vox
    else:
        pixsize = (y_vox,x_vox)

    # On the basis that the diffrent field of you are not saved under the same file
    exp_para = dict(x=w_pixel,y=h_pixel,t=i_m['frames'],
                    v = 1,total_images_per_channel=i_m['images'],
                    pixel_microns=pixsize,
                    interval_sec=i_m['finterval'],axes=i_m['axes'])
    
    # Check if the interval is not too long
    if i_m['finterval']>2000: print(f"Warning: {in_var} has a frame interval of {i_m['finterval']}s")
    
    # Get the number of channels and slices
    if 'channels' in i_m: exp_para['true_c'] = i_m['channels']
    else: exp_para['true_c'] = 1
    
    if 'slices' in i_m: exp_para['z'] = i_m['slices']
    else: exp_para['z'] = 1
    
    return exp_para

def get_ND2_meta(in_var): 
    # Get ND2 img metadata
    nd_obj = ND2Reader(in_var)
    
    # Get meta (sizes always include txy)
    exp_para = nd_obj.sizes
    for k,v in nd_obj.metadata.items():
        if k in ['date','fields_of_view','frames','z_levels','total_images_per_channel','channels','pixel_microns',]:
            exp_para[k] = v
    
    if 'c' not in exp_para: exp_para['true_c'] = 1
    else: 
        exp_para['true_c'] = exp_para['c']
        del exp_para['c']
    
    if 'v' not in exp_para: exp_para['v'] = 1
    
    if 'z' not in exp_para: exp_para['z'] = 1
    ### Check for nd2 bugs with foccused EDF and z stack
    if exp_para['z']*exp_para['t']*exp_para['v']!=exp_para['total_images_per_channel']:
        exp_para['z'] = 1
    
    # Get the interval
    ts = np.round(np.diff(nd_obj.timesteps[::exp_para['v']*exp_para['z']]/1000).mean())
    if np.isnan(ts):# if only single Image, set interval 0
        exp_para['interval_sec'] = 0
    else:
        exp_para['interval_sec'] = int(ts)
    return exp_para

def modifed_meta(channel_list,raw_chan_lst,img_path):
    # Extract raw metadata
    if img_path.endswith('.nd2'):
        exp_para = get_ND2_meta(in_var=img_path)
    elif img_path.endswith('.tif','.tiff'):
        exp_para = get_tif_meta(in_var=img_path)
    
    # Setup metadeta and settings
    temp_exp_dict = {}
    
    ### Add channel properties
    exp_para['channel_list'] = channel_list
    exp_para['c'] = len(channel_list)
    if raw_chan_lst:
        chan_lst = raw_chan_lst
        exp_para['true_channel_list'] = raw_chan_lst
    else:
        if exp_para['true_c']>len(channel_list):
            raise AttributeError(f"{img_path} contains {exp_para['true_c']} channels, but only {len(channel_list)} were given.\nPlease, add all the channels labels, in order, in the variable 'true_channel_list'")
        else:
            chan_lst = channel_list
            exp_para['true_channel_list'] = channel_list
    
    # Create an experiment folder for each field of view
    exp_pathS = []
    for serie in range(exp_para['v']):
        # Create subfolder in parent folder to save the image sequence with a serie's tag
        path_split = img_path.split(sep)
        path_split[-1] = path_split[-1].split('.')[0]+f"_s{serie+1}"
        exp_path =  sep.join(path_split)
        if not isdir(exp_path):
            mkdir(exp_path)
        exp_pathS.append(exp_path)
        # Save para
        exp_para['tag1'] = path_split[-3]
        exp_para['tag2'] = path_split[-2]
        exp_para['v_idx'] = serie+1
        exp_prop = {'metadata':exp_para.copy(),'img_preProcess':{'bg_sub':'None','reg':False,'blur':False},
        'fct_inputs':{},'masks_process':{}}
        temp_exp_dict[exp_path] = exp_prop
    
    return chan_lst,exp_pathS,temp_exp_dict

def name_img(exp_para):
    # Create a name for each image
    img_name = []
    for t in range(exp_para['t']):
        for z in range(exp_para['z']):
            for chan in exp_para['channel_list']:
                img_name.append(chan+'_f%04d'%(t+1)+'_z%04d'%(z+1))
    return img_name

def write_ND2(img_path,img_name,v,exp_para,im_folder):
    # Open ND2 file
    nd_obj = ND2Reader(img_path)
    
    # Unpack img_name
    t,z = [int(i[1:])-1 for i in img_name.split('_')[1:]]             
    c = exp_para['true_channel_list'].index(img_name.split('_')[0])
    
    # Get the image       
    if exp_para['z']>1: 
        temp_img = nd_obj.get_frame_2D(c=c,t=t,z=z,x=exp_para['x'],y=exp_para['y'],v=v-1)
    else: temp_img = nd_obj.get_frame_2D(c=c,t=t,x=exp_para['x'],y=exp_para['y'],v=v-1)
    # Save
    imwrite(join(sep,im_folder+sep,img_name)+".tif"),temp_img.astype(np.uint16)
    
def write_tif(img_path,img_name,exp_para,im_folder):
    # Open tif file
    # Add missing axes
    pass



# def write_tif(img_path,img_name,exp_para,im_folder):
#     # Open tif file
#     img = imread(img_path)
#     axes = exp_para['axes'] # img.shape will be in the order TZCYX
    
#     # Unpack img_name
#     t,z = [int(i[1:])-1 for i in img_name.split('_')[1:]]
#     c = exp_para['true_channel_list'].index(img_name.split('_')[0])
    
#     # Get the image
#     temp_img = img[t,z,...]
#     for t in range(exp_para['t']):
#         if 'T' in axes: temp_img = img[t,...]
        
#         for z in range(exp_para['z']):
#             if 'Z' in axes: temp_img = temp_img[z,...]
            
#             for chan in channel_list:
#                 c = raw_chan_lst.index(chan)
#                 if 'C' in axes: temp_img = temp_img[c,...]
#                 # Save img
#                 imwrite(join(sep,im_folder+sep,chan+'_f%04d'%(t+1)+'_z%04d'%(z+1)+".tif"),temp_img.astype(np.uint16))

def create_imseq(img_path,imseq_ow,channel_list,raw_chan_lst): #TODO: modify pickle as json
    # Get ND2 meta
    chan_lst,exp_pathS,temp_exp_dict = modifed_meta(channel_list=channel_list,raw_chan_lst=raw_chan_lst,img_path=img_path)
    exp_dict = {} # Create empty dict
    
    ### Create the image sequence
    for exp_path in exp_pathS:
        # Create a folder to store all the images
        im_folder = join(sep,exp_path+sep,'Images')
        if not isdir(im_folder):
            mkdir(im_folder)
    
        # If imseq exists just load stack. Else re-/create imseq
        if exists(join(sep,exp_path+sep,'REMOVED_EXP.txt')):
                print(f"-> Exp.: {exp_path} has been removed\n")
                continue
        
        if any(scandir(im_folder)) and not imseq_ow:
            # Log
            print(f"-> Image sequence already exists for exp.: {exp_path}")
            # Update exp_dict
            if not exists(join(sep,exp_path+sep,'exp_properties.pickle')):
                # If old folder without exp_properties, then add it
                temp_exp_dict[exp_path]['status'] = 'active'
                Utility.save_settings(exp_path=exp_path,exp_prop=temp_exp_dict[exp_path])
            else: exp_dict[exp_path] = Utility.open_exp_prop(exp_path=exp_path)
            continue
        
        # Log
        print(f"-> Image sequence is being created for exp.: {exp_path}")

        # Remove all files from dir if ow, to avoid clash with older file version
        if imseq_ow:
            for files in sorted(listdir(im_folder)):
                remove(join(sep,im_folder+sep,files))
        
        # Load exp_para
        exp_prop = temp_exp_dict[exp_path]
        exp_para = exp_prop['metadata']
        exp_prop['status'] = 'active'
        
        if img_path.endswith('.nd2'):
            write_ND2(img_path=img_path,channel_list=channel_list,exp_para=exp_para,chan_lst=chan_lst,im_folder=im_folder)
        elif img_path.endswith('.tif','.tiff'):
            write_tif(img_path=img_path,channel_list=channel_list,exp_para=exp_para,chan_lst=chan_lst,im_folder=im_folder)
    
        # Update exp_dict
        exp_prop['fct_inputs']['create_imseq'] = {'img_path':img_path,'imseq_ow':imseq_ow,'channel_list':channel_list,'raw_chan_lst':raw_chan_lst,}
        exp_dict[exp_path] = exp_prop
        Utility.save_settings(exp_path=exp_path,exp_prop=exp_prop)
    return exp_dict

class Utility():
    #### Utility for creating image seq
    @staticmethod
    def man_bg_sub(imgFold_path): # TODO: use draw poly instead and run as batch
        print(f"--> Applying 'Manual' background substraction on {imgFold_path}")
        
        # Get the exp_path and load exp_para
        exp_path = sep.join(imgFold_path.split(sep)[:-1])
        exp_prop = Utility.open_exp_prop(exp_path=exp_path)
        exp_para = exp_prop['metadata']
        
        # Load stack
        img_stack = Utility.load_stack(imgFold_path=imgFold_path)
        
        # Convert to 3 rgb-like images
        img_stack = Utility.multichan_stack(img_stack=img_stack,exp_para=exp_para)

        # Pre-process stack
        if exp_para['z'] > 1: temp_stack = np.amax(img_stack,axis=1)
        else: temp_stack = img_stack.copy()
        
        if exp_para['t'] > 1: temp_stack = np.amax(temp_stack,axis=0)
        
        # Check that only 3 channels are loaded and convert to bgr.
        if temp_stack.shape[-1]>3: temp_stack = temp_stack[...,:3]
        temp_stack = temp_stack[...,::-1]

        # Get the roi for bg
        roi = draw_rect(temp_stack.astype(np.uint8),imgFold_path.split(sep)[-1])

        # Create a rectangle to apply to each image or not if 'skip' was pressed
        if roi:
            # Rescale rect
            scale = exp_para['x'] / 768 # 768 was used in draw_rect()
            for i, item in enumerate(roi):
                cor_x, cor_y = item
                roi[i] = (round(cor_x*scale),round(cor_y*scale))
            # Get the rectangle
            row, col = draw.rectangle(start=roi[0],end=roi[1],shape=(exp_para['x'],exp_para['y']))

            # Apply the rectangle, determine the mean of the area and remove it from the whole frame
            for n_chan, chan in enumerate(exp_para['channel_list']):
                for f in range(exp_para['t']):
                    # Build image names
                    frame_name = '_f%04d' % (f+1)
                    for z in range(exp_para['z']):
                        # Build image names
                        z_name = '_z%04d'%(z+1)
                        # Get frame and convert to 32 bit
                        if exp_para['t'] > 1:
                            if exp_para['z']==1: im = img_stack[f,...,n_chan].astype(np.float32)
                            else: im = img_stack[f,z,...,n_chan].astype(np.float32)
                        else:
                            if exp_para['z']==1: im = img_stack[...,n_chan].astype(np.float32)
                            else: im = img_stack[z,...,n_chan].astype(np.float32)
                        # Get mean of the bg as rounded to closest intenger
                        bg = int(np.rint(np.mean(im[col,row])))
                        # Remove bg and reset neg val to 0
                        im -= bg
                        im[im < 0] = 0
                        # Resave stack
                        imwrite(join(sep,imgFold_path+sep,chan+frame_name+z_name+'.tif'),im.astype(np.uint16))
            
            # Save pickle
            exp_prop['fct_inputs']['man_bg_sub'] = {'imgFold_path':imgFold_path}
            exp_prop['img_preProcess']['bg_sub'] = 'Manual'
            Utility.save_settings(exp_path=exp_path,exp_prop=exp_prop)
        return exp_prop
    
    @staticmethod
    def smo_bg_sub(imgFold_path,sigma=0.0,size=7):
        # Log
        print(f"--> Applying 'Auto' background substraction on {imgFold_path} with: sigma={sigma} and size={size}")
        
        # Get the exp_path and load exp_para
        exp_path = sep.join(imgFold_path.split(sep)[:-1])
        exp_prop = Utility.open_exp_prop(exp_path=exp_path)
        exp_para = exp_prop['metadata']

        # Initiate SMO
        smo = SMO(shape=(exp_para['x'],exp_para['y']),sigma=sigma,size=size)

        # Run one image at a time
        for file in sorted(listdir(imgFold_path)):
            if file.endswith('.tif'):
                # Load image
                img = imread(join(sep,imgFold_path+sep,file))
                # Get mean of the bg as rounded to closest intenger
                bg_img = smo.bg_corrected(img)
                # Reset neg val to 0
                bg_img[bg_img<0] = 0
                # Resave stack
                imwrite(join(sep,imgFold_path+sep,file),bg_img.astype(np.uint16))
        
        # Save pickle
        exp_prop['fct_inputs']['smo_bg_sub'] = {'imgFold_path':imgFold_path,'smo_sigma':sigma,'smo_size':size}
        exp_prop['img_preProcess']['bg_sub'] = 'Auto'
        Utility.save_settings(exp_path=exp_path,exp_prop=exp_prop)
        return exp_prop
    
    #### Utility for pre-processing images
    @staticmethod
    def im_reg(imgFold_path,reg_channel,reg_mtd,chan_shift,reg_ref,reg_ow):
        # Get the exp_path and load exp_para
        exp_path = sep.join(imgFold_path.split(sep)[:-1])
        exp_prop = Utility.open_exp_prop(exp_path=exp_path)
        exp_para = exp_prop['metadata']

        # Check if time sequence
        if exp_para['t'] == 1:
            print(f"--> {imgFold_path} CANNOT be registrated as it is NOT a time sequence")
            exp_prop['img_preProcess']['reg'] = False
        else:
            exp_prop['img_preProcess']['reg'] = True
            
            # Create save dir if need be
            reg_im_path = imgFold_path+'_Registered'
            if not isdir(reg_im_path):
                mkdir(reg_im_path)
            # Check if images exist
            if any(scandir(reg_im_path)) and not reg_ow:
                # Log
                print(f"--> {imgFold_path} are already registered")
            else:
                # Check that ref and method are correct
                ref_list = ['first','previous','mean']
                if reg_ref not in ref_list:
                    raise ValueError(f"{reg_ref} is not valid. Please only put {ref_list}")
                mtd_list = ['translation','rigid_body','scaled_rotation','affine','bilinear']
                if reg_mtd not in mtd_list:
                    raise ValueError(f"{reg_mtd} is not valid. Please only put {mtd_list}")  
                print(f"--> Registering {imgFold_path} with method '{reg_mtd}' and reference '{reg_ref}'")        
                
                # Load method
                if reg_mtd=='translation':       sr = StackReg(StackReg.TRANSLATION)
                elif reg_mtd=='rigid_body':      sr = StackReg(StackReg.RIGID_BODY)
                elif reg_mtd=='scaled_rotation': sr = StackReg(StackReg.SCALED_ROTATION)
                elif reg_mtd=='affine':          sr = StackReg(StackReg.AFFINE)
                elif reg_mtd=='bilinear':        sr = StackReg(StackReg.BILINEAR)

                # Apply chan_shift
                if chan_shift:
                    print("---> Applying channel shift first")
                    for f in range(exp_para['t']):
                        # Load ref
                        ref = Utility.load_stack(imgFold_path=imgFold_path,channel_list=reg_channel,input_range=[f])
                        # Load im
                        frame_name = '_f%04d'%(f+1)
                        for chan in exp_para['channel_list']:
                            im = Utility.load_stack(imgFold_path=imgFold_path,channel_list=chan,input_range=[f])
                            for z in range(exp_para['z']):
                                # Build z name
                                z_name = '_z%04d.tif'%(z+1)
                                # Apply transfo
                                if exp_para['z']>1: reg_im = sr.register_transform(ref[z,...],im[z,...])
                                else: reg_im = sr.register_transform(ref,im)
                                # Save
                                reg_im[reg_im<0] = 0
                                imwrite(join(sep,reg_im_path+sep,chan+frame_name+z_name),reg_im.astype(np.uint16))

                if reg_ref=='first':
                    # Load ref image
                    if exp_para['z']==1: ref = Utility.load_stack(imgFold_path=imgFold_path,channel_list=reg_channel,input_range=[0])
                    else:                ref = np.amax(Utility.load_stack(imgFold_path=imgFold_path,channel_list=reg_channel,input_range=[0]),axis=0)
                    for f in range(exp_para['t']):
                        # Load image to register
                        im = Utility.load_stack(imgFold_path=imgFold_path,channel_list=reg_channel,input_range=[f])
                        # Get the transfo matrix
                        if exp_para['z']==1: tmats = sr.register(ref=ref,mov=im)
                        else:                tmats = sr.register(ref=ref,mov=np.amax(im,axis=0))
                        # Build image names
                        frame_name = '_f%04d'%(f+1)
                        for chan in exp_para['channel_list']:
                            # Load image to transform
                            im = Utility.load_stack(imgFold_path=imgFold_path,channel_list=chan,input_range=[f])
                            for z in range(exp_para['z']):
                                # Build z name
                                z_name = '_z%04d.tif'%(z+1)
                                # Apply transfo
                                if exp_para['z']==1: reg_im = sr.transform(mov=im,tmat=tmats)
                                else: reg_im = sr.transform(mov=im[z,...],tmat=tmats)
                                # Save
                                reg_im[reg_im<0] = 0
                                imwrite(join(sep,reg_im_path+sep,chan+frame_name+z_name),reg_im.astype(np.uint16))
                elif reg_ref=='mean':
                    # Load ref image
                    if exp_para['z']==1: ref = np.mean(Utility.load_stack(imgFold_path=imgFold_path,channel_list=reg_channel),axis=0)
                    else:                ref = np.mean(np.amax(Utility.load_stack(imgFold_path=imgFold_path,channel_list=reg_channel),axis=1),axis=0)

                    for f in range(exp_para['t']):
                        # Load image to register
                        im = Utility.load_stack(imgFold_path=imgFold_path,channel_list=reg_channel,input_range=[f])
                        # Get the transfo matrix
                        if exp_para['z']==1: tmats = sr.register(ref=ref,mov=im)
                        else:                tmats = sr.register(ref=ref,mov=np.amax(im,axis=0))
                        # Build image names
                        frame_name = '_f%04d'%(f+1)
                        for chan in exp_para['channel_list']:
                            # Load image to transform
                            im = Utility.load_stack(imgFold_path=imgFold_path,channel_list=chan,input_range=[f])
                            for z in range(exp_para['z']):
                                # Build z name
                                z_name = '_z%04d.tif'%(z+1)
                                # Apply transfo
                                if exp_para['z']==1: reg_im = sr.transform(mov=im,tmat=tmats)
                                else: reg_im = sr.transform(mov=im[z,...],tmat=tmats)
                                # Save
                                reg_im[reg_im<0] = 0
                                imwrite(join(sep,reg_im_path+sep,chan+frame_name+z_name),reg_im.astype(np.uint16))
                elif reg_ref=='previous':
                    for f in range(1,exp_para['t']):
                        # Load ref image
                        if f==1:
                            if exp_para['z']==1: ref = Utility.load_stack(imgFold_path=imgFold_path,channel_list=reg_channel,input_range=[f-1])
                            else:                ref = np.amax(Utility.load_stack(imgFold_path=imgFold_path,channel_list=reg_channel,input_range=[f-1]),axis=0)
                        else:
                            if exp_para['z']==1: ref = Utility.load_stack(imgFold_path=reg_im_path,channel_list=reg_channel,input_range=[f-1])
                            else:                ref = np.amax(Utility.load_stack(imgFold_path=reg_im_path,channel_list=reg_channel,input_range=[f-1]),axis=0)
                        # Load image to register
                        im = Utility.load_stack(imgFold_path=imgFold_path,channel_list=reg_channel,input_range=[f])
                        # Get the transfo matrix
                        if exp_para['z']==1: tmats = sr.register(ref=ref,mov=im)
                        else:                tmats = sr.register(ref=ref,mov=np.amax(im,axis=0))
                        # Build image names
                        frame_name = '_f%04d'%(f+1)
                        for chan in exp_para['channel_list']:
                            # Load image to transform
                            im = Utility.load_stack(imgFold_path=imgFold_path,channel_list=chan,input_range=[f])
                            for z in range(exp_para['z']):
                                # Build z name
                                z_name = '_z%04d.tif'%(z+1)
                                # Copy the first image to the reg_folder
                                if f==1:
                                    fst = Utility.load_stack(imgFold_path=imgFold_path,channel_list=chan,input_range=[f-1])
                                    if exp_para['z']==1: imwrite(join(sep,reg_im_path+sep,chan+'_f0001'+z_name),fst.astype(np.uint16))
                                    else: imwrite(join(sep,reg_im_path+sep,chan+'_f0001'+z_name),fst[z,...].astype(np.uint16))
                                # Apply transfo
                                if exp_para['z']==1: reg_im = sr.transform(mov=im,tmat=tmats)
                                else: reg_im = sr.transform(mov=im[z,...],tmat=tmats)
                                # Save
                                reg_im[reg_im<0] = 0
                                imwrite(join(sep,reg_im_path+sep,chan+frame_name+z_name),reg_im.astype(np.uint16))

                # Update settings
                exp_prop['fct_inputs']['im_reg'] = {'imgFold_path':imgFold_path,
                                                    'reg_channel':reg_channel,'reg_mtd':reg_mtd,
                                                    'reg_ref':reg_ref,'reg_ow':reg_ow}
                if 'channel_seg' in exp_prop:
                    if 'Images_Registered' in exp_prop['channel_seg']:
                        if reg_channel not in exp_prop['channel_seg']['Images_Registered']:
                            exp_prop['channel_seg']['Images_Registered'].append(reg_channel)
                    else:
                        exp_prop['channel_seg']['Images_Registered'] = [reg_channel]
                else:
                    exp_prop['channel_seg'] = {'Images_Registered':[reg_channel]}
                
        # Save settings
        Utility.save_settings(exp_path=exp_path,exp_prop=exp_prop)
        return exp_prop

    @staticmethod
    def blur_img(imgFold_path,blur_kernel,blur_sigma,blur_ow):
        # Check if input are correct type
        if type(blur_kernel)!=tuple:
            raise TypeError("The input 'blur_kernel' must be a tuple: (int,int)")

        # Check if kernel contains 2 odd intengers >= to 3
        if not all(i%2!=0 for i in blur_kernel) and not all(i>=3 for i in blur_kernel):
            print("The input 'blur_kernel' must contain 2 odd intengers greater or equal to 3")

        # Get the exp_path and load exp_para
        exp_path = sep.join(imgFold_path.split(sep)[:-1])
        exp_prop = Utility.open_exp_prop(exp_path=exp_path)

        # Create blur dir
        blur_img_path = join(sep,exp_path+sep,'Images_Blured')
        if not isdir(blur_img_path):
            mkdir(blur_img_path)
        
        # Check if exist
        if any(scandir(blur_img_path)) and not blur_ow:
             # Log
            print(f"-> Images are already blured with a kernel of {exp_prop['fct_inputs']['blur_img']['blur_kernel']} and sigma of {exp_prop['fct_inputs']['blur_img']['blur_sigma']}")
        else:
            # Log
            print(f"-> Bluring images using a kernel of {blur_kernel} and sigma of {blur_sigma}")
            for file in sorted(listdir(imgFold_path)):
                # Load image
                if file.endswith('.tif'): im = imread(files=join(sep,imgFold_path+sep,file))
                # Blur image and save
                imwrite(join(sep,blur_img_path+sep,file),cv2.GaussianBlur(im,blur_kernel,blur_sigma).astype(np.uint16))
            
            # Save settings
            exp_prop['fct_inputs']['blur_img'] = {'imgFold_path':imgFold_path,'blur_kernel':blur_kernel,'blur_sigma':blur_sigma,'blur_ow':blur_ow}
            exp_prop['img_preProcess']['blur'] = True
            Utility.save_settings(exp_path=exp_path,exp_prop=exp_prop)
        return exp_prop,blur_img_path
    
    @staticmethod
    def thresholding(img,thres,):
        # Set the threshold's value. Either as input or automatically if thres==None
        if thres:
            retVal = thres
        else:
            retVal,__ = cv2.threshold(img.astype(np.uint8),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
        # Apply the threshold
        __,temp = cv2.threshold(img.astype(np.uint16),retVal,255,cv2.THRESH_BINARY)
        
        # Clean the mask of small holes and small objects
        mask = remove_small_holes(temp.astype(bool),50)
        return remove_small_objects(mask,1000).astype(bool),retVal

    #### Utility for loading images/masks/settings
    @staticmethod# 
    def load_stack(imgFold_path,channel_list=None,input_range=None):
        # Get the exp_path and load exp_para
        exp_path = sep.join(imgFold_path.split(sep)[:-1])
        exp_prop = Utility.open_exp_prop(exp_path=exp_path)
        exp_para = exp_prop['metadata']

        # Check for channel
        if channel_list:
            if type(channel_list)==str:
                chan_lst = [channel_list]
            else:
                chan_lst = channel_list
        else:
            chan_lst = exp_para['channel_list']

        if input_range:
            frame_range = input_range
        else:
            frame_range = range(exp_para['t'])

        # Load/Reload stack. Expected shape of images tzxyc
        exp_list = []
        for chan in chan_lst:
            chan_list = []
            for frame in frame_range:
                f_lst = []
                for im in sorted(listdir(imgFold_path)):
                    # To be able to load either _f3digit.tif or _f4digit.tif
                    ndigit = len(im.split('_')[-1].split('.')[0][1:])
                    if im.startswith(chan) and im.__contains__(f'_f%0{ndigit}d'%(frame+1)):
                        f_lst.append(imread(join(sep,imgFold_path+sep,im)))
                chan_list.append(f_lst)
            exp_list.append(chan_list)
        if len(chan_lst)==1:
            stack = np.squeeze(np.stack(exp_list))
        else:
            stack = np.moveaxis(np.squeeze(np.stack(exp_list)), [0], [-1])
        return stack
    
    @staticmethod
    def get_batch(img_path,ram_ratio,chan_numb=1,z_size=1,frames=None):
        # Ram_ratio: cp = 17, bax = 1.2
        
        # Size of a frame
        if isinstance(img_path,str):
            frame_size = getsize(img_path)*ram_ratio*chan_numb*z_size
        elif isinstance(img_path,list):
            frame_size = getsize(img_path[0])*ram_ratio*chan_numb*z_size

        # Number of frames that will take ~80% of RAM available
        max_frames_allowed = ceil(virtual_memory().available*0.8/frame_size)
        
        if frames:
            # Calculate number of batches needed
            batch_numb = ceil(frames/max_frames_allowed)
            # Get the batches
            batches = np.array_split(np.array(range(frames)),batch_numb)
        else:
            batch_numb = ceil(len(img_path)/max_frames_allowed)
            batches = np.array_split(np.array(range(len(img_path))),batch_numb)

        if batch_numb==1:
            print('Loading the entire stack')
        else:
            print(f"Loading stack into {batch_numb} batches")
            
        # Convert batch to list:
        batches = [list(batch) for batch in batches]
        return batches
    
    @staticmethod
    def load_mask(maskFold_path,channel_seg,mask_shape=None,input_range=None,do_log=True):
        if not any(channel_seg in file for file in listdir(maskFold_path)):
            raise AttributeError(f'No masks was found for the selected segmented channel: {channel_seg}')
        
        # Get the exp_path and load exp_para
        exp_path = sep.join(maskFold_path.split(sep)[:-1])
        exp_prop = Utility.open_exp_prop(exp_path=exp_path)
        exp_para = exp_prop['metadata']

        if input_range:
            if not isinstance(input_range,list):
                raise TypeError("input range must be a list")
            frame_range = input_range
        else:
            frame_range = list(range(exp_para['t']))

        # Load masks
        if mask_shape:
            if type(mask_shape)!=str:
                raise TypeError(f"mask_shape must be a string")

        if do_log:
            if mask_shape: print(f"---> Loading frames {frame_range} of {mask_shape} masks from {maskFold_path}")
            else: print(f"---> Loading frames {frame_range} of masks from '{maskFold_path}'")

        # load mask files
        exp_list = []
        for f in frame_range:
            for im in sorted(listdir(maskFold_path)):
                # To be able to load either _f3digit.tif or _f4digit.tif
                ndigit = len(im.split('_')[-1].split('.')[0][1:])
                if mask_shape:
                    if im.__contains__(f'mask_%s%s_f%0{ndigit}d'%(channel_seg,mask_shape,f+1)):
                        exp_list.append(imread(join(sep,maskFold_path+sep,im)))
                else:
                    if im.__contains__(f'mask_%s_f%0{ndigit}d'%(channel_seg,f+1)):
                        exp_list.append(imread(join(sep,maskFold_path+sep,im)))
        return np.squeeze(np.stack(exp_list))
    
    @staticmethod
    def save_mask(mask_stack,save_dir,batch,z_size,frames,chan_seg,motif=None):
        # Convert np.array to list of arr
        if isinstance(mask_stack,np.ndarray):
            if mask_stack.ndim==2 or frames==1:
                mask_stack = [mask_stack]
            else:
                mask_stack = [m for m in mask_stack]
        
        if motif:
            motif = '_'+motif
        else:
            motif = ''

        for i,f in enumerate(batch):
            f_name = '_f%04d'%(f+1)
            for z in range(z_size):
                z_name = '_z%04d'%(z+1)
                if z_size==1:
                    imwrite(join(sep,save_dir+sep,f'mask_{chan_seg}{motif}'+f_name+z_name+'.tif'),mask_stack[i].astype(np.uint16))
                else:
                    imwrite(join(sep,save_dir+sep,f'mask_{chan_seg}{motif}'+f_name+z_name+'.tif'),mask_stack[i][z,...].astype(np.uint16))
    
    @staticmethod
    def multichan_stack(img_stack,exp_para):
        """This function convert image stack (np.array) into a multichannel stack (at least 3 channels) if needed.
        If the experiment contains less than 3 channels, then blank(s) channel(s) will be added. 

        Args:
            img_stack (np.array): Image stack to convert if needed to multichannel stack.
            sequence_length (int): Length of the image sequence.
            x_size (int): Size of x axis.
            y_size (int): Size of y axis.

        Returns:
            (np.array): Returns a multichannel stack."""
            
        # Assess the number of channel to add or not blank ones
        if exp_para['c'] == 1:
            # Add dim
            img_stack = img_stack[...,np.newaxis]
            # Add empty channel
            if exp_para['t'] > 1:
                if exp_para['z']==1:
                    zerarr = np.zeros((exp_para['t'],exp_para['x'],exp_para['y'],2))
                else:
                    zerarr = np.zeros((exp_para['t'],exp_para['z'],exp_para['x'],exp_para['y'],2))
            else:
                if exp_para['z']==1:
                    zerarr = np.zeros((exp_para['x'],exp_para['y'],2))
                else:
                    zerarr = np.zeros((exp_para['z'],exp_para['x'],exp_para['y'],2))
        elif exp_para['c'] == 2:
            # Add empty channel
            if exp_para['t'] > 1:
                if exp_para['z']==1:
                    zerarr = np.zeros((exp_para['t'],exp_para['x'],exp_para['y'],1))
                else:
                    zerarr = np.zeros((exp_para['t'],exp_para['z'],exp_para['x'],exp_para['y'],1))
            else:
                if exp_para['z']==1:
                    zerarr = np.zeros((exp_para['x'],exp_para['y'],1))
                else:
                    zerarr = np.zeros((exp_para['z'],exp_para['x'],exp_para['y'],1))
            
        # Re-stack if needed
        if exp_para['c'] < 3:
            exp_stack_rgb = np.concatenate([img_stack, zerarr], axis=-1)
        else:
            exp_stack_rgb = img_stack
        return exp_stack_rgb

    @staticmethod
    def get_pos_cell(maskFold_path,frames):
        # Open summary of tracking file and prepare df for positive and negative cells
        track_res = pd.read_csv(join(sep,maskFold_path+sep,'res_track.txt'), 
                                sep=" ", header=None, names=['ID','FF','LF','isPos'])
    
        # List of positive cells
        positive_track = track_res.loc[(track_res['FF']==0)&(track_res['LF']==frames-1),'ID'].tolist()
        return positive_track
    
    @staticmethod
    def trim_mask(stack,positive_track):
        # Trim obj
        for obj in list(np.unique(stack))[1:]:
            if obj not in positive_track:
                stack[stack==obj] = 0
        return stack
    
    @staticmethod
    def open_settings(exp_path):
        with open(join(sep,exp_path+sep,"exp_settings.json")) as fp:
            exp_prop = json.load(fp)
        return exp_prop
    
    @staticmethod #[ ]: Changed settings to json
    def save_settings(exp_path,exp_prop):
        with open(join(sep,exp_path+sep,"exp_settings.json"), "w") as fp:
            json.dump(obj=exp_prop,fp=fp,indent=4)

    #### Utility for post-processing masks
    @staticmethod
    def modif_stitch3D(masks,stitch_threshold):
        # Invert stitch_threshold
        stitch_threshold = 1 - stitch_threshold
        # basic stitching from Cellpose
        masks = stitch3D(masks, stitch_threshold=stitch_threshold)
        
        # create mastermask to have all possible cells on one mask. Doing this by doing 'mode' operation 
        # to get the value present in most t-frames per pixel. Ignoring backgound by setting zero to nan. Threfore conversion to float is needed.
        rawmasks_ignorezero = masks.copy().astype(float)
        rawmasks_ignorezero[rawmasks_ignorezero == 0] = np.nan
        master_mask = mode(rawmasks_ignorezero, axis=0, keepdims=False, nan_policy='omit')[0]
        master_mask = master_mask.astype(int)

        # second stitch round by using mastermask to compair with every frame
        #slighly changed code from Cellpose 'stitch3D'
        """ stitch 2D masks into 3D volume with stitch_threshold on IOU """
        mmax = masks[0].max()
        empty = 0

        for i in range(len(masks)):
            iou = _intersection_over_union(masks[i], master_mask)[1:,1:]
            if not iou.size and empty == 0:
                mmax = masks[i].max()
            elif not iou.size and not empty == 0:
                icount = masks[i].max()
                istitch = np.arange(mmax+1, mmax + icount+1, 1, int)
                mmax += icount
                istitch = np.append(np.array(0), istitch)
                masks[i] = istitch[masks[i]]
            else:
                iou[iou < stitch_threshold] = 0.0
                iou[iou < iou.max(axis=0)] = 0.0
                istitch = iou.argmax(axis=1) + 1
                ino = np.nonzero(iou.max(axis=1)==0.0)[0]
                istitch[ino] = 0
                istitch = np.append(np.array(0), istitch)
                masks[i] = istitch[masks[i]]
                empty = 1
        return masks
    
    @staticmethod
    def morph(mask_stack,n_mask=2):
        """
        This function will patch incomplete tracks. It will try to connect cells with a maximum gap determined by deltaT, only if their overlap is higher than 
        minOverlay. If the resulting tracks has missing frames, they will then be warped. All other non-linked traks will be discarded.

        Args:
            stack (np.array): Masks array generated by either BaxTrack or Stitching
            path (str): Path where the tracked masks are saved.
            deltaT (int): Define the maximum gap between cells to be linked.
            minOverlay (float): Define the minimum overlay ratio between cells to be linked. Threshold should be within 0 to 1.
            reg (bool): Are images registered or not (i.e. there's no shift/movement between frames). It will influence patching.

        Returns:
            (np.array): patched masks array
            ([int]): new list of positive tracks
        """
        n_stack = np.zeros((mask_stack.shape))
        for obj in list(np.unique(mask_stack))[1:]:
            temp = mask_stack.copy()
            temp[temp!=obj] = 0
            framenumber = len(np.unique(np.where(mask_stack == obj)[0]))
            if framenumber!=mask_stack.shape[0] and framenumber > n_mask:
                temp = fill_gaps(temp)
            n_stack = n_stack + temp
            if np.any(n_stack>obj):
                n_stack[n_stack>obj] = n_stack[n_stack>obj]-obj
        # Recheck for incomplete track
        for obj in list(np.unique(n_stack))[1:]:
            fn = len(np.unique(np.where(n_stack==obj)[0]))
            if fn!=mask_stack.shape[0]:
                n_stack[n_stack==obj] = 0
        n_stack = n_stack.astype('uint16') 
        return n_stack

    @staticmethod
    def erode_mask(mask_stack,exp_para,rad_ero,**kwargs):
        # Set up erode
        pat_ero = disk(rad_ero)
        mask_eroded = np.zeros((mask_stack.shape))

        # Run erosion
        for obj in list(np.unique(mask_stack))[1:]:
            temp = mask_stack.copy()
            temp[temp!=obj] = 0
            for f in range(exp_para['t']):
                for z in range(exp_para['z']):
                    if exp_para['t']==1:
                        if exp_para['z']==1:
                            ero = cv2.erode(temp,pat_ero,**kwargs)
                            mask_eroded += ero
                        else:
                            ero = cv2.erode(temp[z,...],pat_ero,**kwargs)
                            mask_eroded[z,...] += ero
                    else:
                        if exp_para['z']==1:
                            ero = cv2.erode(temp[f,...],pat_ero,**kwargs)
                            mask_eroded[f,...] += ero
                        else:
                            ero = cv2.erode(temp[f,z,...],pat_ero,**kwargs)
                            mask_eroded[f,z,...] += ero
        return mask_eroded

    @staticmethod
    def dilate_mask(mask_stack,exp_para,rad_dil,**kwargs):
        # Set up erode
        pat_dil = disk(rad_dil)
        mask_dilated = np.zeros((mask_stack.shape))

        # Run erosion
        for f in range(exp_para['t']):
            for z in range(exp_para['z']):
                if exp_para['t']==1:
                    if exp_para['z']==1:
                        dil = cv2.dilate(src=mask_stack,kernel=pat_dil,**kwargs)
                        mask_dilated += dil
                    else:
                        dil = cv2.dilate(src=mask_stack[z,...],kernel=pat_dil,**kwargs)
                        mask_dilated[z,...] += dil
                else:
                    if exp_para['z']==1:
                        dil = cv2.dilate(src=mask_stack[f,...],kernel=pat_dil,**kwargs)
                        mask_dilated[f,...] += dil
                    else:
                        dil = cv2.dilate(src=mask_stack[f,z,...],kernel=pat_dil,**kwargs)
                        mask_dilated[f,z,...] += dil
        return mask_dilated

    @staticmethod
    def ref_mask(imgFold_path,maskLabel,ref_mask_ow): #TODO if there is bf, use bf for the maskdrawing (except for maybe laser wounds?)
        """
        This function first generate wound mask on input images (it uses the same channel as segmentation).
        Then mask is converted into distance transformed. Finally, using centroids of masks, we determine the
        position of each cells regarding the wound.
        """
        # Check masklabel
        if type(maskLabel)!=str:
            raise TypeError(f"Maskname cannot be of type {type(maskLabel)}. Only string is accepted")
        
        # Get the exp_path and load exp_para
        exp_path = sep.join(imgFold_path.split(sep)[:-1])
        exp_prop = Utility.open_exp_prop(exp_path=exp_path)
        exp_para = exp_prop['metadata']

        # Create mask_ref dir
        mask_ref_path = join(sep,exp_path+sep,'Masks_'+maskLabel)
        if not isdir(mask_ref_path):
            mkdir(mask_ref_path)

        if any(scandir(mask_ref_path)) and not ref_mask_ow:
            mask_ref = Utility.load_mask(maskFold_path=mask_ref_path,channel_seg=maskLabel,do_log=False)
            
            print(f"-> Loading reference {'Masks_'+maskLabel} masks from Exp. {exp_path}")
        else:
            # Get get masks for dmap
            if exp_para['t'] == 1:
                mask_stack = Utility.multichan_stack(img_stack=Utility.load_stack(imgFold_path=imgFold_path),exp_para=exp_para)
                if exp_para['z']>1: mask_stack = np.amax(a=mask_stack,axis=0)
                mask_ref = get_pre_wound(mask_stack)
                mask_name = f"mask_{maskLabel}_f0001.tif"
                imwrite(join(sep,mask_ref_path+sep,mask_name),mask_ref.astype(np.uint16))
            else:
                mask_stack = Utility.multichan_stack(img_stack=Utility.load_stack(imgFold_path=imgFold_path),exp_para=exp_para)
                if exp_para['z']>1: mask_stack = np.amax(a=mask_stack,axis=1)
                mask_ref = fill_gaps(get_pre_wound(mask_stack))
                for m in range(exp_para['t']):
                    mask_name = f"mask_{maskLabel}_f%04d.tif" % (m + 1)
                    imwrite(join(sep,mask_ref_path+sep,mask_name),mask_ref[m,...].astype(np.uint16))
        return mask_ref
    
    @staticmethod
    def apply_dmap(mask_stack,frames):
        if frames==1:
            dt_mask = mh.distance(np.logical_not(mask_stack),metric='euclidean')    
        else:
            dt_mask = np.stack([mh.distance(np.logical_not(mask_stack[m,...]),metric='euclidean') for m in range(frames)])
        return dt_mask

    @staticmethod
    def pixel_bin(df_pixel,intBin,col_name,deltaF):
        # Determine the number of bins
        nbin = np.arange(start=0,stop=df_pixel['dmap'].max()+intBin,step=intBin)
        df_pixel.loc[:,'bin'] = pd.cut(df_pixel.loc[:,'dmap'],bins=nbin)

        # Rearrange the data to mean per bin
        bin_df = df_pixel.groupby(['time','bin']).mean().reset_index()
        bin_df = bin_df.pivot('time','bin',col_name).T
        bin_df['bin_start'] = 0
        for idx in bin_df.index:
            bin_df.loc[idx,'bin_start'] = int(idx.left)
        bin_df.set_index('bin_start',inplace=True)

        # Apply normlisation?
        if deltaF:
            if not isinstance(deltaF,list) or len(deltaF)!=2 or all(not isinstance(x,Number) for x in deltaF):
                raise AttributeError(f"The variable 'deltaF' must be a list of 2 integer/float")
            col_index = bin_df.columns[(bin_df.columns>=deltaF[0])&(bin_df.columns<=deltaF[1])]
            f0 = bin_df.loc[:,col_index].mean(axis=1,skipna=True)
            bin_df = bin_df.subtract(f0,axis=0).divide(f0,axis=0)
        return bin_df.fillna(0)

    @staticmethod
    def plot_HM(df,title,axes=None,figsize=None,savedir=None,cbar_label=r'$\Delta$F/F$_{min}$',col_lim=[0,2],**kwargs):
        # Get kwargs for sb.heatmap
        plot_args = {'cmap':'jet','yticklabels':10,'xticklabels':5,'cbar_kws':{'label': cbar_label},
                        'vmin':col_lim[0],'vmax':col_lim[1]}
        plot_args.update(kwargs)
        
        # plot
        fig = plt.figure(figsize=figsize)
        plt.rcParams['pdf.fonttype'] = 42
        sb.set_style()
        ax = sb.heatmap(df,ax=axes,**plot_args)
        ax.set_title(title)
        ax.set_xlabel('Time (min)')
        ax.set_ylabel(r'Distance from wound ($\mu$m)')
        ax.invert_yaxis()
        
        if savedir:
            fig = ax.get_figure()
            fig.savefig(join(sep,savedir+sep,f'{title}.pdf'))
        return ax

    @staticmethod
    def get_ratio(lst):
        combi = []
        for chan in lst:
            t_lst = lst.copy(); t_lst.remove(chan)
            combi += [(chan,x) for x in t_lst]
        return combi
    
    @staticmethod
    def transfo_df(df,channel_list,stim_time=None,start_baseline=0,posCont_time=None,signal_type='linear'): #TODO: Check deltaF
        # Check input
        if signal_type not in ['linear','ocsillatory']: raise AttributeError(f"signal_type can only be {['linear','ocsillatory']}")
        
        # Shortcut
        cl = 'condition_label';bl = 'basal';st = 'stimulus';pc = 'positive_control'
        pcd = 'perCondition'

        # Apply all possible ratio
        pair_lst = Utility.get_ratio(channel_list)
        for c1,c2 in pair_lst:
            df[f"{c1}/{c2}"] = df[c1]/df[c2]
        df.replace([np.inf, -np.inf], 0, inplace=True)
        df.fillna(0)
        
        # Add 'condition_label'
        df[cl] = 'other'
        if stim_time: 
            df.loc[(df['time']>=start_baseline)&(df['time']<stim_time),cl] = bl
            df.loc[df['time']>=stim_time,cl] = st
        if posCont_time: df.loc[df['time']>=posCont_time,cl] = pc

        # Apply all possible deltaF
        if stim_time:
            deltaF_lst = channel_list+[f"{c1}/{c2}" for c1,c2 in pair_lst]
            bi_lst = deltaF_lst+[f'deltaF_{k}' for k in deltaF_lst]
            new_col = [f"deltaF_{k}" for k in deltaF_lst]+[f"{col}_{pcd}" for col in bi_lst]
            df = df.reindex(columns=df.columns.to_list()+new_col,fill_value=0)
            for c in df['cell'].unique():
                dft = df.loc[(df['cell']==c)].copy()
                # Apply all possible deltaF
                for col_delta in deltaF_lst:
                    f0 = dft.groupby(cl).mean().loc[bl,col_delta]
                    # f0 = float(dft.loc[df[cl]==bl,col_delta].mean())
                    if posCont_time: 
                        # fmax_val = float(dft.loc[df[cl]==pc,col_delta].max())
                        fmax_val = dft.groupby(cl).max().loc[pc,col_delta]
                        perf0 = fmax_val-f0
                    else: perf0 = f0
                    df0 = (dft[col_delta]-f0)/perf0
                    df.loc[(df['cell']==c),f'deltaF_{col_delta}'] = df0.values
                # Add all condition value
                for col in bi_lst:
                    if signal_type=='linear':
                        df.loc[(df['cell']==c)&(df[cl]==bl),f"{col}_{pcd}"] = df.loc[(df['cell']==c)&(df[cl]==bl),col].mean()
                        df.loc[(df['cell']==c)&(df[cl]==st),f"{col}_{pcd}"] = df.loc[(df['cell']==c)&(df[cl]==st),col].mean()
                        df.loc[(df['cell']==c)&(df[cl]==pc),f"{col}_{pcd}"] = df.loc[(df['cell']==c)&(df[cl]==pc),col].mean()
                    else:
                        df.loc[(df['cell']==c)&(df[cl]==bl),f"{col}_{pcd}"] = df.loc[(df['cell']==c)&(df[cl]==bl),col].median()
                        df.loc[(df['cell']==c)&(df[cl]==st),f"{col}_{pcd}"] = df.loc[(df['cell']==c)&(df[cl]==st),col].median()
                        df.loc[(df['cell']==c)&(df[cl]==pc),f"{col}_{pcd}"] = df.loc[(df['cell']==c)&(df[cl]==pc),col].median()
        return Utility.get_outliers(df=df.copy(),col_lst=channel_list,signal_type=signal_type)

    @staticmethod
    def get_outliers(df,col_lst,signal_type):
        col_keys = ['z-score','z-outlier','q1','q3','iqr','q1*iqr','q3*iqr','iqr-outlier']
        new_col = [f"{k}_{col}" for col in col_lst for k in col_keys]
        # Shortcuts
        cl = 'condition_label'
        fs = 'frames'
        # Add all columns
        for col in col_lst:
            df.insert(df.columns.get_loc(col)+1,f"z-score_{col}",0)
            df.insert(df.columns.get_loc(col)+2,f"z-outlier_{col}",0)
            df.insert(df.columns.get_loc(col)+3,f"q1_{col}",0)
            df.insert(df.columns.get_loc(col)+4,f"q3_{col}",0)
            df.insert(df.columns.get_loc(col)+5,f"iqr_{col}",0)
            df.insert(df.columns.get_loc(col)+6,f"q1*iqr_{col}",0)
            df.insert(df.columns.get_loc(col)+7,f"q3*iqr_{col}",0)
            df.insert(df.columns.get_loc(col)+8,f"iqr-outlier_{col}",0)
        
        # Compute outliers
        if signal_type=='linear':
            for f in df[fs].unique():
                for col in col_lst:
                    # compute z-score
                    df.loc[df[fs]==f,f"z-score_{col}"] = np.abs(stats.zscore(df.loc[df[fs]==f,col]))
                    # compute iqr
                    df.loc[df[fs]==f,f"q1_{col}"] = df.loc[df[fs]==f,col].quantile(q=.25)
                    df.loc[df[fs]==f,f"q3_{col}"] = df.loc[df[fs]==f,col].quantile(q=.75)
                    df.loc[df[fs]==f,f"iqr_{col}"] = iqr(x=df.loc[df[fs]==f,col])
                    df.loc[df[fs]==f,f"q1*iqr_{col}"] = df.loc[df[fs]==f,f"q1_{col}"]-1.5*df.loc[df[fs]==f,f"iqr_{col}"]
                    df.loc[df[fs]==f,f"q3*iqr_{col}"] = df.loc[df[fs]==f,f"q3_{col}"]+1.5*df.loc[df[fs]==f,f"iqr_{col}"]
        elif signal_type=='ocsillatory':
            meddf = df.groupby(['condition_label','cell']).median().reset_index()
            for cond in df['condition_label'].unique():
                for col in col_lst:
                    # Compute z-score
                    meddf.loc[meddf['condition_label']==cond,f"z-score_{col}"] = np.abs(stats.zscore(meddf.loc[meddf['condition_label']==cond,col]))
                    # Compute iqr
                    meddf.loc[meddf['condition_label']==cond,f"q1_{col}"] = meddf.loc[meddf['condition_label']==cond,col].quantile(q=.25)
                    meddf.loc[meddf['condition_label']==cond,f"q3_{col}"] = meddf.loc[meddf['condition_label']==cond,col].quantile(q=.75)
                    meddf.loc[meddf['condition_label']==cond,f"iqr_{col}"] = iqr(meddf.loc[meddf['condition_label']==cond,col])
                    meddf.loc[meddf['condition_label']==cond,f"q1*iqr_{col}"] = meddf.loc[meddf['condition_label']==cond,f"q1_{col}"]-1.5*meddf.loc[meddf['condition_label']==cond,f"iqr_{col}"]
                    meddf.loc[meddf['condition_label']==cond,f"q3*iqr_{col}"] = meddf.loc[meddf['condition_label']==cond,f"q3_{col}"]+1.5*meddf.loc[meddf['condition_label']==cond,f"iqr_{col}"]
            
            meddf = meddf.set_index(['condition_label','cell'])
            for ind,row in df.iterrows():
                cell = row.cell; cond = row.condition_label
                for col in new_col:
                    row[col] = meddf.loc[(cond,cell),col]
                df.loc[ind] = row
            
        # Label outliers
        for c in df.cell.unique():
            for col in col_lst:
                # Z-score decision
                if any(df.loc[df.cell==c,f"z-score_{col}"]>=3): df.loc[df.cell==c,f"z-outlier_{col}"] = 1
                else: df.loc[df.cell==c,f"z-outlier_{col}"] = 0
                # iqr decision
                if any(df.loc[df.cell==c,col]<df.loc[df.cell==c,f"q1*iqr_{col}"])|any(df.loc[df.cell==c,col]>df.loc[df.cell==c,f"q3*iqr_{col}"]):
                    df.loc[df.cell==c,f"iqr-outlier_{col}"] = 1
                else: df.loc[df.cell==c,f"iqr-outlier_{col}"] = 0
        
        return df

##############################
#### Utility fct
def draw_rect(img,img_name): # TODO: to be deleted
    """Draw a rectangle on loaded image and get roi as output.
    Take as input single 3 channels image (shape=x,y,c).
    Usage:
        - The image will appear in popup window
        - Select the area of your choice. 
        - Then press 's' (i.e. "save"), when you're satisfy
        - Alternativelly, press 'q' (i.e. "quit"), if you don't want to process a particuliar stack

    Args:
        - img (np.array): Single 3 channels image in uint8.
        - path (str): Path of the loaded image.
        
    Returns:
        - refPt ([tup]): List of 2 tuples. Each tuples are the coordinate x and y of the drawn rectangle (e.g. [(x0, y0), (x1, y1)])"""
    
    # Create variable
    refPt = [] # record coordinate of start and end to draw rectangle
    tempPt = [] # record temporary end coordinate to be able to update rectangle as mouse
    drawing = False # to keep track on when to draw
    
    def draw_rectangle_with_drag(event, x, y, flags, param):
        # nonlocal: to be able to access the variable generated in function above
        nonlocal refPt, tempPt, drawing, im

        # Create an event when press the left mouse button
        if event == cv2.EVENT_LBUTTONDOWN: 
            # get the starting coordinate
            refPt = [(x, y)]            
            drawing = True
            
            # copy the image to always reset the rectangle after each clic
            im = im2.copy()
            
        # Create an event when mouse is moving
        elif event == cv2.EVENT_MOUSEMOVE:
            # Record temporary end coordinate so the rectangle update as it goes
            tempPt = [(x, y)]

        # Create an event when release the mouse button
        elif event == cv2.EVENT_LBUTTONUP:
            # record the final coordinate
            refPt.append((x, y))
            drawing = False
            
            # Draw rectangle on image
            cv2.rectangle(im, refPt[0], refPt[1], color =(0, 255, 255), thickness = 1)

    # Create popup window with mouse event embeded
    cv2.namedWindow(f"Select ROI for {img_name}")
    cv2.setMouseCallback(f"Select ROI for {img_name}", draw_rectangle_with_drag)
    
    # Open the image and make a copy
    im = cv2.resize(img,(768,768),cv2.INTER_NEAREST)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im2 = im.copy()
    
    # Add all the text
    textc = "Press 's' to save ROI"
    coordc = (10,20)
    texts = "Press 'q' to skip experiment"
    coords = (10,40)
    
    font = cv2.FONT_HERSHEY_PLAIN
    fontScale = 1.2
    color = (0,255,255)
    thickness = 1
    
    cv2.putText(im, textc, coordc, font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.putText(im, texts, coords, font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.putText(im2, textc, coordc, font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.putText(im2, texts, coords, font, fontScale, color, thickness, cv2.LINE_AA)

    # Loop to keep track of the drawing
    while True:
        if not drawing:
            # When static, show the image (with/without the rectangle)
            cv2.imshow(f"Select ROI for {img_name}", im)
            
        elif drawing and tempPt:
            # When drawing, update the rectangle in real time
            rect_copy = im.copy()
            cv2.rectangle(rect_copy, refPt[0], tempPt[0], (0, 255, 255), 1)
            cv2.imshow(f"Select ROI for {img_name}", rect_copy)

        # Create a 'key' event 
        key = cv2.waitKey(1) & 0xFF
        
        # Press a key to terminate the loop
        if key == ord("s"): # press 'c' to confirm and get refPt selected
            break
            
        elif key == ord("q"): # press 's' to skip bg from the stack
            refPt = []
            break
            
    # Close all windows
    cv2.destroyAllWindows()
    
    # Get the roi as output
    return refPt

def bbox_ND(array):
    """
    This function take a np.array (any dimension) and create a bounding box around the nonzero shape.
    Also return a slice object to be able to reconstruct to the originnal shape.

    Args:
        array (np.array): Array containing a single mask. The array can be of any dimension.

    Returns:
        (tuple): Tuple containing the new bounding box array and the slice object used for the bounding box. 
    """
    # Determine the number of dimensions
    N = array.ndim
    
    # Go trhough all the axes to get min and max coord val
    slice_list = []
    for ax in itertools.combinations(reversed(range(N)), N - 1):
        nonzero = np.any(array, axis=ax)
        vmin, vmax = np.where(nonzero)[0][[0, -1]]
        # Store these coord as slice obj
        slice_list.append(slice(vmin,vmax+1))
    
    s = tuple(slice_list)
    
    return tuple([array[s], s])

def mask_warp(m1,m2,ngap):
    def center_mask(mask,midX,midY):
        # Get centroid of mask
        M1 = cv2.moments(mask)
        cX = int(M1["m10"] / M1["m00"])
        cY = int(M1["m01"] / M1["m00"])

        # Get interval of centroid
        Yint = midY-cY
        Xint = midX-cX

        Ys,Xs = np.where(mask!=0)
        
        # Check that it stays within borders of array
        nY = Ys+Yint
        nY[nY<0] = 0
        nY[nY>mask.shape[0]-1] = mask.shape[0]-1
        nY = nY.astype(int)
        
        nX = Xs+Xint
        nX[nX<0] = 0
        nX[nX>mask.shape[1]-1] = mask.shape[1]-1
        nX = nX.astype(int)
        
        # Move the obj
        n_masks = np.zeros((mask.shape))
        obj_val = int(list(np.unique(mask))[1])
        for points in list(zip(nY,nX)):
            n_masks[points] = obj_val
        return n_masks,(cY,cX)
    
    # Get middle of array
    midX = int(m1.shape[1]/2)
    midY = int(m1.shape[0]/2)

    n1,c1 = center_mask(m1,midX,midY)
    n2,c2 = center_mask(m2,midX,midY)

    # Centroids linespace
    Xs = np.linspace(c1[1],c2[1],ngap+2)
    Ys = np.linspace(c1[0],c2[0],ngap+2)

    overlap, crop_slice = bbox_ND(n1+n2)

    # Crop and get the overlap of both mask
    #overlap,crop_slice = bbox_ND((m1+m2))
    n1_cropped = n1[crop_slice]
    n2_cropped = n2[crop_slice]
    overlap[overlap!=np.max(n1)+np.max(n2)] = 0

    # Get the ring (i.e. non-overlap area of each mask)
    ring1 = n1_cropped+overlap
    ring1[ring1!=np.max(n1_cropped)] = 0
    ring2 = (n2_cropped+overlap)
    ring2[ring2!=np.max(n2_cropped)] = 0

    if np.any(ring1!=0) or np.any(ring2!=0):  #check for different shapes, otherwise just copy shape
        # Get the distance transform of the rings with overlap as 0 (ref point)
        # dt = distance_transform_bf(np.logical_not(overlap))
        dt = distance(np.logical_not(overlap),metric='euclidean')
        dt1 = dt.copy()
        dt2 = dt.copy()
        dt1[ring1==0] = 0
        dt2[ring2==0] = 0

        # Create the increment for each mask, i.e. the number of step needed to fill the gaps
        # if max == 0, then it means that mask is completly incorporated into the other one and will have no gradient

        max1 = np.max(dt1)
        if max1 != 0:
            inc1 = list(np.linspace(max1, 0, ngap+1, endpoint=False)) # First mask gets chewed-in
            inc1.pop(0)

        max2 = np.max(dt2)
        if max2 != 0:
            inc2 = list(np.linspace(0, max2, ngap+1, endpoint=False)) # Second mask gets bigger
            inc2.pop(0) 

        # Fill the gaps
        masks_list = []
        for i in range(ngap):
            # Select part of the mask that falls out and reset pixel vals to 1        
            if max1 == 0:
                mA = overlap.copy()
            else:
                mA = dt1.copy()
                mA[mA>inc1[i]] = 0
                mA = mA+overlap
            mA[mA!=0] = 1

            # Select part of the mask that are added and reset pixel vals to 1
            if max2 == 0:
                mB = overlap.copy()
            else:
                mB = dt2.copy()
                mB[mB>inc2[i]] = 0
                mB = mB+overlap
            mB[mB!=0] = 1

            # Recreate the full shape
            mask = mA+mB
            mask[mask!=0] = np.max(n1)

            # Resize the mask
            resized_mask = np.zeros((m1.shape))
            resized_mask[crop_slice] = mask

            # Replace mask to new center position
            resized_mask,__ = center_mask(mask=resized_mask,midX=np.round(Xs[i+1]),midY=np.round(Ys[i+1]))

            # append the list
            masks_list.append(resized_mask)
    else:
        # Fill the gaps
        masks_list = []
        for i in range(ngap):
            mask = overlap.copy()

            # Resize the mask
            resized_mask = np.zeros((m1.shape))
            resized_mask[crop_slice] = mask

            # Replace mask to new center pisotion
            resized_mask,__ = center_mask(mask=resized_mask,midX=np.round(Xs[i+1]),midY=np.round(Ys[i+1]))

            # append the list
            masks_list.append(resized_mask)
    return masks_list

def fill_gaps(stack): # TODO: make this faster
    """
    This function determine how many missing frames (i.e. empty frames, with no masks) there are from a stack.
    It will then fill the gaps using mask_warp().

    Args:
        stack (np.array): Mask array with missing frames. 

    Returns:
        stack (np.array): Mask array with filled frames.
    """
    # Find the frames with masks and without for a given obj: bool
    is_masks = [np.any(i) for i in stack]
    # Identify masks that suround empty frames
    masks_loc = []
    for i in range(len(is_masks)):
        if (i == 0) or (i == len(is_masks)-1):
            if is_masks[i]==False:
                masks_loc.append(0)
            else:
                masks_loc.append(1)
        else:
            if (is_masks[i]==True) and (is_masks[i-1]==False):
                masks_loc.append(1) # i.e. first mask after empty
            elif (is_masks[i]==True) and (is_masks[i+1]==False):
                masks_loc.append(1) # i.e. last mask before empty
            elif (is_masks[i]==True) and (is_masks[i-1]==True):
                masks_loc.append(2) # already filled with masks
            elif (is_masks[i]==False):
                masks_loc.append(0)

    # Get the key masks
    masks_id = [i for i in range(len(masks_loc)) if masks_loc[i] == 1]

    # Copy the first and/or last masks to the ends of the stacks if empty
    stack[:masks_id[0],...] = stack[masks_id[0],...]
    stack[masks_id[-1]:,...] = stack[masks_id[-1],...]

    # Get the indexes of the masks to morph (i.e. that suround empty frames) and the len of empty gap
    masks_to_morph = []
    for i in range(len(masks_id)-1):
        if any([i in [0] for i in masks_loc[masks_id[i]+1:masks_id[i+1]]]):
            masks_to_morph.append([masks_id[i],masks_id[i+1],len(masks_loc[masks_id[i]+1:masks_id[i+1]])])

    # Morph and fill stack
    for i in masks_to_morph:
        n_masks = mask_warp(stack[i[0]],stack[i[1]],i[2])
        stack[i[0]+1:i[1],...] = n_masks
    return stack

def draw_polygons(img):
    """
    Free-hand draw a polygon on loaded image. Take as input list of path files of the images.
    Usage:
        - The image will appear in popup window
        - Press either 'b' or 'f' to move frames back- or forward
        - Draw polygon
        - Press 's' to save polygon. If a polygon is drawn and save on the same frame, it will overwrite the previous one.
        - Press 'q' when ready

    Args:
        img_list ([str]): List of path file of the images.
        
    Returns:
        dict_roi (dict): Dictionary containg the coordinates of the polygons drawn for each selected frames. 
    """
    # Determine sequence length
    if img.ndim==4:
        seqLeng = img.shape[0]
    elif img.ndim==3:
        seqLeng = 1
    
    # Load images and draw
    f = 0 # Allow to move between the different frames
    dict_roi = {} # Store the polygons
    img2  = img.copy()
    alpha = 1; beta = 0
    togglemask = 0; togglelabel = 0
    conbri = 0
    while f!=-1:
        drawing=False; polygons = []; currentPt = []
        # Mouse callback function
        def freehand_draw(event,x,y,flags,param):
            nonlocal polygons, drawing, im, currentPt
            # Press mouse
            if event==cv2.EVENT_LBUTTONDOWN:
                drawing=True; polygons = []; currentPt = []
                currentPt.append([x,y])
                polygons = np.array([currentPt], np.int32)
                im = im2.copy()
            # Draw when mouse move, if pressed
            elif event==cv2.EVENT_MOUSEMOVE:
                if drawing==True:
                    cv2.polylines(im,[polygons],False,(0,255,255),2)
                    currentPt.append([x,y])
                    polygons = np.array([currentPt], np.int32)
            # Release mouse button
            elif event==cv2.EVENT_LBUTTONUP:
                drawing=False
                cv2.polylines(im,[polygons],True,(0,255,255),2)
                cv2.fillPoly(im,[polygons],(0,255,255))
                currentPt.append([x,y])
                polygons = np.array([currentPt], np.int32)
            return polygons
        
        # Read/Load image
        if seqLeng==1:
            im = cv2.resize(img,(768,768),cv2.INTER_NEAREST)
        else:
            im = cv2.resize(img[f],(768,768),cv2.INTER_NEAREST)
        im2 = im.copy()

        if togglemask == 0:
            if f in dict_roi.keys():
                cv2.polylines(im,dict_roi[f], True, (0,255,255),2)
                cv2.fillPoly(im,dict_roi[f],(0,255,255))

        cv2.namedWindow("Draw ROI of the Wound")
        cv2.setMouseCallback("Draw ROI of the Wound",freehand_draw)
        
        # Setup labels
        text = f"Frame {f+1}/{seqLeng}"; coord = (320,20)
        texts = "Press 's' to save ROI and move forward"; coords = (10,40)
        if system()=='Linux': textar = "Press 'l' to go forward"
        elif system()=='Windows': textar = "Press 'ARROW RIGHT' to go forward"
        coordar = (10,60)
        if system()=='Linux': textal = "Press 'j' to go backward"
        elif system()=='Windows': textal = "Press 'ARROW LEFT' to go backward"
        coordal = (10,80)
        textq = "Press 'q' for quit"; coordq = (10,100)
        if system()=='Linux': textc = "Press 'c' once and 'i'(up) or 'k'(down) to change contrast"
        elif system()=='Windows': textc = "Press 'c' once and 'ARROW UP/DOWN' to change contrast"
        coordc = (10,120)
        if system()=='Linux': textb = "Press 'b' once and 'i' or 'k' to change brightness"
        elif system()=='Windows': textb = "Press 'b' once and 'ARROW UP/DOWN' to change brightness"
        coordb = (10,140)
        textx = "Press 'x' to toggle mask"; coordx = (10,160)  
        textl = "Press 'h' to toggle help"; coordl = (10,180)                  

        font = cv2.FONT_HERSHEY_PLAIN; fontScale = 1.2; color = (0,255,255); thickness = 1
        
        # Apply label on images
        cv2.putText(im, text, coord, font, fontScale, color, thickness, cv2.LINE_AA)
        cv2.putText(im2, text, coord, font, fontScale, color, thickness, cv2.LINE_AA)
        if togglelabel == 0:
            cv2.putText(im, textal, coordal, font, fontScale, color, thickness, cv2.LINE_AA)
            cv2.putText(im, textar, coordar, font, fontScale, color, thickness, cv2.LINE_AA)
            cv2.putText(im, texts, coords, font, fontScale, color, thickness, cv2.LINE_AA)
            cv2.putText(im, textq, coordq, font, fontScale, color, thickness, cv2.LINE_AA)
            cv2.putText(im, textc, coordc, font, fontScale, color, thickness, cv2.LINE_AA)
            cv2.putText(im, textb, coordb, font, fontScale, color, thickness, cv2.LINE_AA)
            cv2.putText(im, textx, coordx, font, fontScale, color, thickness, cv2.LINE_AA)
            cv2.putText(im, textl, coordl, font, fontScale, color, thickness, cv2.LINE_AA)

            cv2.putText(im2, textal, coordal, font, fontScale, color, thickness, cv2.LINE_AA)
            cv2.putText(im2, textar, coordar, font, fontScale, color, thickness, cv2.LINE_AA)
            cv2.putText(im2, texts, coords, font, fontScale, color, thickness, cv2.LINE_AA)
            cv2.putText(im2, textq, coordq, font, fontScale, color, thickness, cv2.LINE_AA)
            cv2.putText(im2, textc, coordc, font, fontScale, color, thickness, cv2.LINE_AA)
            cv2.putText(im2, textb, coordb, font, fontScale, color, thickness, cv2.LINE_AA)
            cv2.putText(im2, textx, coordx, font, fontScale, color, thickness, cv2.LINE_AA)
            cv2.putText(im2, textl, coordl, font, fontScale, color, thickness, cv2.LINE_AA)
        
        while True:
            cv2.imshow("Draw ROI of the Wound",im)

            # Numbers for arrow keys
            # Windows: left: 2424832, right: 2555904, up: 2490368, down: 2621440
            # Linux: left: 65361, right: 65363, up: 65362, down: 65364

            key = cv2.waitKeyEx(1)
            
            # if key != -1:
            #     print(key)
            
            # press 'q' to exit.
            if key == ord("q"):
                f = -1
                conbri = 0
                break
            # press 'arrow key right' to move forward
            elif key == 2555904 or key == ord("l"): #ArrowKey RIGHT for Windows
                conbri = 0
                if f == seqLeng-1:
                    f = seqLeng-1
                    break
                else:
                    f += 1
                    break
            # press 'arrow key left' to move backwards
            elif key == 2424832 or key == ord("j"): #ArrowKey LEFT for Windows
                conbri = 0
                if f == 0:
                    f = 0
                    break
                else:
                    f -= 1
                    break
            # press 's' to save roi.
            elif key == ord("s"):
                conbri = 0
                if f == seqLeng-1:
                    dict_roi[f] = polygons
                    f = seqLeng-1
                    break
                else:
                    dict_roi[f] = polygons
                    f += 1
                    break   
            # press 'c' to activate contrast change mode
            elif key == ord("c"):
                conbri = key
            # press 'b' to activate brightness change mode
            elif key == ord("b"):
                conbri = key
            # if Arrowkey up or down is pressed, check if contrast or brightness change mode is active
            elif key == 2490368 or key == ord("i"): #ArrowKey UP for Windows
                if conbri == ord("c"):
                    alpha += 5
                    img = cv2.convertScaleAbs(img2, alpha=alpha, beta=beta)
                    break
                elif conbri == ord("b"):
                    beta += 5
                    img = cv2.convertScaleAbs(img2, alpha=alpha, beta=beta)  
                    break 
            elif key ==  2621440 or key == ord("k"): #ArrowKey DOWN for Windows
                if conbri == ord("c"):
                    alpha += -5
                    if alpha < 1:
                        alpha = 1
                    img = cv2.convertScaleAbs(img2, alpha=alpha, beta=beta)
                    break
                elif conbri == ord("b"):
                    beta += -5
                    img = cv2.convertScaleAbs(img2, alpha=alpha, beta=beta)
                    break   
            # toogle masks visibility
            elif key == ord("x"):
                if togglemask == 0:
                    togglemask = 1
                    break
                else:
                    togglemask = 0
                    break
            # toogle label visibility
            elif key == ord("h"):
                if togglelabel == 0:
                    togglelabel = 1
                    break
                else:
                    togglelabel = 0     
                    break
    cv2.destroyAllWindows()
    return dict_roi

def get_pre_wound(img):
    """This function will load images to draw the wound. The output will be np.array of the mask

    Args:
        img (np.array): Image stack in rgb.

    Returns:
        pre_wound (np.array): Mask array with gaps.
    """
    # Make single channel image RGB-like
    # if img.ndim==2:
    #     img = ((img - img.min()) * (1/(img.max() - img.min()) * 255)).astype(np.uint8)
    #     img = cv2.merge([img,img,img])

    # Determine the dimension of input img
    if img.ndim == 4:
        y = img.shape[1]
        x = img.shape[2]
        # Create binary image to create pre_wound array
        pre_wound = np.zeros(img.shape[:3])
    elif img.ndim == 3:
        y = img.shape[0]
        x = img.shape[1]
        # Create binary image to create pre_wound array
        pre_wound = np.zeros(img.shape[:2])
    
    # Check that only 3 channels are loaded and convert to bgr
    if img.shape[-1]>3:
        img = img[...,:3]
    img = img[...,::-1]

    # Draw polygons
    poly_dict = draw_polygons(img.astype(np.uint8))

    # Iterate through dict
    for k, v in poly_dict.items():
        t_im = np.zeros((768,768),np.uint8)
        t_im = cv2.fillPoly(t_im,[v],1)
        t_im = cv2.resize(t_im,(y,x),cv2.INTER_NEAREST)
        if img.ndim == 4:
            pre_wound[k,...] = t_im.astype(np.uint8)
        else:
            pre_wound = t_im.astype(np.uint8)
    return pre_wound


