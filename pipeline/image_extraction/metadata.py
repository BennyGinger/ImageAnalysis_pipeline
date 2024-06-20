from __future__ import annotations
from os import sep, mkdir, PathLike
from os.path import isdir
from tifffile import TiffFile
from nd2 import ND2File
import warnings

def get_tif_meta(img_path: PathLike) -> dict:
    tiff_meta = {}
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

    if 'finterval' not in imagej_meta: 
        imagej_meta['finterval'] = None
        print("Warning: No frame interval found in metadata. Defaulting to None")
    else:
        imagej_meta['finterval'] = int(imagej_meta['finterval'])

    original_keys = ['ImageWidth','ImageLength','frames','channels','slices','axes','finterval']
    new_keys = ['img_width','img_length','n_frames','full_n_channels','n_slices','axes','interval_sec']

    # Rename meta
    for i, key in enumerate(original_keys):
        tiff_meta[new_keys[i]] = imagej_meta[key]

    # Uniformize meta
    tiff_meta['n_series'] = 1
    tiff_meta['um_per_pixel'] = calculate_um_per_pixel(imagej_meta)
    tiff_meta['file_type'] = '.tif'
    return tiff_meta

def calculate_um_per_pixel(meta_dict: dict) -> tuple[float,float] | None:
    """Calculate the um per pixel from the metadata of a tiff file. Output axes = (x,y)"""
    # Check if resolution was extracted
    if 'XResolution' not in meta_dict or 'YResolution' not in meta_dict:
        print("Warning: Resolution not found in metadata. Defaulting to None")
        return None
    
    # Calculate um per pixel
    x_um_per_pix = round(1/meta_dict['XResolution'],ndigits=3)
    y_um_per_pix = round(1/meta_dict['YResolution'],ndigits=3)
    return x_um_per_pix,y_um_per_pix

def get_ND2_meta(img_path: PathLike)-> dict:
    # Get ND2 img metadata
    with ND2File(img_path) as nd_obj:
        nd2meta = {**nd_obj.sizes}
        # Add missing keys and get axes
        nd2meta['axes'] = 'TPZCYX'
        if 'T' not in nd2meta:
            nd2meta['T'] = 1
            nd2meta['axes'] = nd2meta['axes'].replace('T','')
        if 'C' not in nd2meta:
            nd2meta['C'] = 1
            nd2meta['axes'] = nd2meta['axes'].replace('C','')  
        if 'Z' not in nd2meta:
            nd2meta['Z'] = 1
            nd2meta['axes'] = nd2meta['axes'].replace('Z','')   
        if 'P' not in nd2meta:
            nd2meta['P'] = 1
            nd2meta['axes'] = nd2meta['axes'].replace('P','')

        # Rename meta
        original_keys = ['C', 'Z', 'T', 'P', 'X', 'Y']
        new_keys = ['full_n_channels', 'n_slices', 'n_frames', 'n_series', 'img_width', 'img_length']
        for i, key in enumerate(original_keys):
            nd2meta[new_keys[i]] = nd2meta.pop(key)

        # Uniformize meta
        nd2meta['um_per_pixel'] = nd_obj.metadata.channels[0].volume.axesCalibration[:2]
        if nd2meta['n_frames']>1:
            if nd_obj.experiment[0].type == 'TimeLoop': #for normal timelapse experiments
                nd2meta['interval_sec'] = nd_obj.experiment[0].parameters.periodMs/1000
            elif nd_obj.experiment[0].type == 'NETimeLoop': #for ND2 Merged Experiments
                nd2meta['interval_sec'] = nd_obj.experiment[0].parameters.periods[0].periodMs/1000
        else:
            nd2meta['interval_sec'] = None
        nd2meta['file_type'] = '.nd2'
        nd_obj.close()
    return nd2meta

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
    return meta_dict

def update_channel_names(meta_dict: dict, full_channel_list: list=None, active_channel_list: list=None) -> dict:
    if not full_channel_list:
        warnings.warn(f"\n---- Warning: No exp channels given. All channels will be automatically named ----")
        meta_dict['full_channel_list'] = meta_dict['active_channel_list'] = [f'C{i+1}' for i in range(meta_dict['full_n_channels'])]
        return meta_dict
    
    if len(full_channel_list) != meta_dict['full_n_channels']:
        raise ValueError(f"Number of channels in the image file ({meta_dict['full_n_channels']}) is different from the number of channels given ({len(full_channel_list)})")
    
    # Format the channel names and add them to the meta
    meta_dict['full_channel_list'] = [format_label_name(label) for label in full_channel_list]
    
    if not active_channel_list:
        meta_dict['active_channel_list'] = meta_dict['full_channel_list']
        return meta_dict
    
    # Check if the active channels are in the full channel list
    for channel in active_channel_list:
        if channel not in full_channel_list:
            raise ValueError(f"Channel {channel} not found in the full channel list")
    
    # Format the channel names and add them to the meta
    meta_dict['active_channel_list'] = [format_label_name(label) for label in active_channel_list]
    return meta_dict

def format_label_name(label: str) -> str:
    """Format the channel label name to be used in the pipeline. Must NOT contain any '_'."""
    return label.replace('_','')

# # # # # # # # main functions # # # # # # # # # 
def get_metadata(img_path: PathLike, active_channel_list: list=None, full_channel_list: list=None)-> dict:
    """Gather metadata from all image files (.nd2 and/or .tif) and is attributed to its own experiment folder"""
    
    print(f"\n-> Extracting metadata from \033[94m{img_path}\033[0m")
    if img_path.endswith('.nd2'):
        meta_dict = get_ND2_meta(img_path)
    elif img_path.endswith(('.tif','.tiff')):
        meta_dict = get_tif_meta(img_path)
    else:
        raise ValueError('Image format not supported, please use .nd2 or .tif/.tiff')
    # meta_dict = uniformize_meta(meta_dict)
    
    meta_dict['img_path'] = img_path
    
    meta_dict = create_exp_folder(meta_dict)
    
    # Add channel data
    meta_dict = update_channel_names(meta_dict,active_channel_list,full_channel_list)
    return meta_dict
    
# Final output: 
# {'active_channel_list': ['C1', 'C2'],
#  'axes': 'TZCYX',
#  'exp_path_list': ['/home/Test_images/nd2/Run2/c2z25t23v1_nd2_s1'], return a list pf path based on the number of series
#  'file_type': '.nd2',
#  'full_channel_list': ['C1', 'C2'],
#  'full_n_channels': 2,
#  'img_length': 512,
#  'img_path': '/home/Test_images/nd2/Run2/c2z25t23v1_nd2.nd2',
#  'img_width': 512,
#  'interval_sec': 11,
#  'level_0_tag': 'Run2',
#  'level_1_tag': 'nd2',
#  'n_frames': 23,
#  'n_series': 1,
#  'n_slices': 25,
#  'um_per_pixel': (0.322, 0.322)}
    

if __name__ == '__main__':
    from time import time
    from pathlib import Path
    
    # Test
    t1 = time()
    img_paths = list(Path("/home/Test_images/szimi/MET/20240515-fMLF_diffusion").glob("**/*.tif"))
    img_paths = [str(p) for p in img_paths]
    
    for path in img_paths:
        metadata = get_metadata(path)
        print(metadata['um_per_pixel'])
    
    # with TiffFile(img_paths[0]) as tif:
    #     imagej_meta = tif.imagej_metadata
    #     imagej_meta['axes'] = tif.series[0].axes
    #     for page in tif.pages: # Add additional meta
    #         print(page.tags)
    #         for tag in page.tags:
    #             print(tag)
    #             if tag.name in ['ImageWidth','ImageLength',]:
    #                 imagej_meta[tag.name] = tag.value
    #             if tag.name in ['XResolution','YResolution']:
    #                 imagej_meta[tag.name] = tag.value[0]/tag.value[1]
    # print(imagej_meta)
    
    
    t2 = time()
    print(f"Time to process: {round(t2-t1,ndigits=3)} sec\n")



