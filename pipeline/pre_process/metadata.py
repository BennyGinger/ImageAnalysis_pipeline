from os import sep, mkdir, PathLike
from os.path import isdir
from tifffile import TiffFile
from nd2reader import ND2Reader
import numpy as np

def get_tif_meta(img_path: PathLike) -> dict:
    # Open tif and read meta
    with TiffFile(img_path) as tif:
        imagej_meta = tif.imagej_metadata
        imagej_meta['axes'] = tif.series[0].axes
        for page in tif.pages: # Add additional meta
            for tag in page.tags:
                if tag.name in ['ImageWidth','ImageLength',]:
                    imagej_meta[tag.name] = tag.value
                if tag.name in ['XResolution','YResolution']:
                    imagej_meta[tag.name] = tag.value[0]/tag.value[1]
    
    if 'frames' not in imagej_meta: imagej_meta['frames'] = 1
    
    if 'channels' not in imagej_meta: imagej_meta['channels'] = 1
    
    if 'slices' not in imagej_meta: imagej_meta['slices'] = 1
    
    if 'finterval' not in imagej_meta: imagej_meta['finterval'] = 0
    
    imagej_meta['n_series'] = 1
    return imagej_meta

def calculate_X_pixmicron(x_resolution: float, img_width: int) -> float:
    width_micron = round(img_width/x_resolution,ndigits=3)
    return round(width_micron/img_width,ndigits=3)

def get_ND2_meta(img_path: PathLike)-> dict: 
    # Get ND2 img metadata
    nd_obj = ND2Reader(img_path)
    
    # Get meta (sizes always include txy)
    nd2_meta = {**nd_obj.metadata,**nd_obj.sizes}
    nd2_meta['timesteps'] = nd_obj.timesteps
    if 'c' not in nd2_meta: nd2_meta['c'] = 1
    
    if 'v' not in nd2_meta: nd2_meta['v'] = 1
    
    if 'z' not in nd2_meta: nd2_meta['z'] = 1
    
    nd2_meta['axes'] = ''
    ### Check for nd2 bugs with foccused EDF and z stack
    if nd2_meta['z']*nd2_meta['t']*nd2_meta['v']!=nd2_meta['total_images_per_channel']:
        nd2_meta['z'] = 1
    return nd2_meta

def calculate_interval_sec(timesteps: list, n_frames: int, n_series: int, n_slices: int) -> int:
    # Calculate the interval between frames in seconds
    if n_frames==1: 
        return 0
    interval_sec = np.round(np.diff(timesteps[::n_series*n_slices]/1000).mean())
    
    if int(interval_sec)==0:
        print("\n---- Warning: The interval between frames could not be retrieved correctly. The interval will be set to 0 ----")
        return 0
    return int(interval_sec)

def uniformize_meta(meta_dict: dict) -> dict:
    # Uniformize both nd2 and tif meta
    uni_meta = {}
    new_keys = ['img_width','img_length','n_frames','full_n_channels','n_slices','n_series','pixel_microns','axes','interval_sec','file_type']
    if meta_dict['file_type']=='.nd2':
        old_keys = ['x','y','t','c','z','v','pixel_microns','axes','missing','file_type']
    elif meta_dict['file_type']=='.tif':
        old_keys = ['ImageWidth','ImageLength','frames','channels','slices','n_series','missing','axes','finterval','file_type']
    
    for new_key,old_key in zip(new_keys,old_keys):
        if new_key=='pixel_microns' and old_key=='missing':
            uni_meta[new_key] = calculate_X_pixmicron(meta_dict['XResolution'],meta_dict['ImageWidth'])
        
        elif new_key=='interval_sec' and old_key=='missing':
            uni_meta[new_key] = calculate_interval_sec(meta_dict['timesteps'],meta_dict['t'],meta_dict['v'],meta_dict['z'])
        
        else: uni_meta[new_key] = meta_dict[old_key]
    
    uni_meta['pixel_microns'] = round(uni_meta['pixel_microns'],ndigits=3)
    uni_meta['interval_sec'] = int(round(uni_meta['interval_sec']))
    return uni_meta

def create_exp_folder(meta_dict: dict) -> dict:
    meta_dict['exp_path_list'] = []
    for serie in range(meta_dict['n_series']):
        # Create subfolder in parent folder to save the image sequence with a serie's tag
        path_split = meta_dict['img_path'].split(sep)
        path_split[-1] = path_split[-1].split('.')[0]+f"_s{serie+1}"
        exp_path =  sep.join(path_split)
        if not isdir(exp_path):
            mkdir(exp_path)
        meta_dict['exp_path_list'].append(exp_path)
        # Get tags
        meta_dict['level_1_tag'] = path_split[-3]
        meta_dict['level_0_tag'] = path_split[-2]
    return meta_dict

# # # # # # # # main functions # # # # # # # # # 
def get_metadata(img_path: PathLike, active_channel_list: list, full_channel_list: list=[])-> dict:
    """Gather metadata from all image files (.nd2 and/or .tif) and is attributed to its own experiment folder"""
    print(f"\nExtracting metadata from images")
    if img_path.endswith('.nd2'):
        meta_dict = get_ND2_meta(img_path)
        meta_dict['file_type'] = '.nd2'
    elif img_path.endswith(('.tif','.tiff')):
        meta_dict = get_tif_meta(img_path)
        meta_dict['file_type'] = '.tif'
    else:
        raise ValueError('Image format not supported, please use .nd2 or .tif/.tiff')
    meta_dict = uniformize_meta(meta_dict)
    
    meta_dict['img_path'] = img_path
    
    meta_dict = create_exp_folder(meta_dict)
    
    # Add channel data
    meta_dict['active_channel_list'] = active_channel_list
    if full_channel_list:
        meta_dict['full_channel_list'] = full_channel_list
    else:
        meta_dict['full_channel_list'] = active_channel_list
    
    return meta_dict

# TODO: check the number of channels in the image file...
    

if __name__ == '__main__':
    from time import time
    # Test
    img_path = '/Users/benhome/BioTool/GitHub/cp_dev/c3z1t1v3s1.tif'
    active_channel_list = ['GFP','RFP','DAPI']
    t1 = time()
    img_meta = get_metadata(img_path,active_channel_list)
    t2 = time()
    print(f"Time to get meta: {t2-t1}")
    print(img_meta)

    img_path2 = '/Users/benhome/BioTool/GitHub/cp_dev/c3z1t1v3.nd2'
    t1 = time()
    img_meta2 = get_metadata(img_path2,active_channel_list=active_channel_list)
    t2 = time()
    print(f"Time to get meta: {t2-t1}")
    print(img_meta2)



