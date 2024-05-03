from __future__ import annotations
from dataclasses import dataclass, fields, field
import json
from os import PathLike, listdir
from os.path import join
import numpy as np
import pandas as pd

@dataclass
class LoadClass:
    def from_dict(self, input_dict: dict)-> dict:
        fieldSet = {f.name for f in fields(self) if f.init}
        filteredArgDict = {k : v for k, v in input_dict.items() if k in fieldSet}
        return self(**filteredArgDict)
    
    def set_attribute(self, attr: str, value: any)-> None:
        for field in fields(self):
            if field.name == attr:
                setattr(self,attr,value)
                return
            # If in nested branch
            nested_branch = getattr(self,field.name)
            # Only check if the nested branch is a class
            if isinstance(nested_branch,LoadClass):
                nested_branch.set_attribute(attr,value)
                return
    
    def init_to_inactive(self)-> None:
        """On initialization, set all branches to inactive, to be turn on later by settings.
        This is to avoid conflicts with some processes that were ran but no longer needed."""
        for branch in fields(self):
            # If bool directly in branch
            if isinstance(getattr(self,branch.name),bool):
                setattr(self,branch.name,False)
            # If nested branch
            nested_branch = getattr(self,branch.name)
            # Only check if the nested branch is a class
            if isinstance(nested_branch,LoadClass):
                nested_branch.init_to_inactive()

    @property
    def analyzed_channels(self) -> dict:
        """Property to get the analyzed channels in the class, i.e. the masks. 
        Return a dictionary with the name of the branch as key and the list of channels as value.
        for example: {'cellpose_seg':['channel1','channel2'],'iou_tracking':['channel3','channel4']},
        where exp contains segmentation and tracking branches for the different listed channels."""
        settings_dict = {}
        for branch in fields(self):
            # Check if branch is dict and not empty
            if isinstance(getattr(self, branch.name), dict) and getattr(self, branch.name):
                settings_dict[branch.name] = list(getattr(self, branch.name).keys())
            # If nested branch
            nested_branch = getattr(self,branch.name)
            # Check if branch is dict and not empty
            if isinstance(nested_branch,LoadClass):
                tmp_dict = nested_branch.analyzed_channels
                settings_dict.update(tmp_dict)
        return settings_dict

@dataclass
class PreProcess(LoadClass):
    # Flag wheather the branches are active or not
    is_frame_reg: bool = False
    is_img_blured: bool = False
    # Settings of active branches
    background_sub: list = field(default_factory=list)
    channel_reg: list = field(default_factory=list)
    frame_reg: list = field(default_factory=list)
    img_blured: list = field(default_factory=list)
    
@dataclass
class Segmentation(LoadClass):
    # Flag whether the branches are active or not
    is_threshold_seg: bool = False
    is_cellpose_seg: bool = False
    # Settings of active branches
    threshold_seg: dict = field(default_factory=dict)
    cellpose_seg: dict = field(default_factory=dict)
    
@dataclass
class Tracking(LoadClass):    
    # Flag wheather the branches are active or not
    is_iou_tracking: bool = False
    is_manual_tracking: bool = False
    is_gnn_tracking: bool = False
    # Settings of active branches
    iou_tracking: dict = field(default_factory=dict)
    manual_tracking: dict = field(default_factory=dict)
    gnn_tracking: dict = field(default_factory=dict)

@dataclass
class ImageProperties(LoadClass):
    """Get metadata from nd2 or tif file, using ND2File or TiffFile and ImageJ"""
    img_width: int = None
    img_length: int = None
    n_frames: int = None
    full_n_channels: int = None
    n_slices: int = None
    n_series: int = None
    img_path: str  = None
    
@dataclass
class Analysis(LoadClass):
    um_per_pixel: tuple[float,float] = None
    interval_sec: float = None
    file_type: str = None
    level_0_tag: str = None
    level_1_tag: str = None

@dataclass
class Experiment(LoadClass):
    exp_path: str
    active_channel_list: list = field(default_factory=list)
    full_channel_list: list = field(default_factory=list)
    img_properties: ImageProperties = field(default_factory=ImageProperties)
    analysis: Analysis = field(default_factory=Analysis)
    preprocess: PreProcess = field(default_factory=PreProcess)
    segmentation: Segmentation = field(default_factory=Segmentation)
    tracking: Tracking = field(default_factory=Tracking)

    @property
    def ori_imgs_lst(self)-> list[PathLike]:
        """Return the list of paths of original images."""
        im_folder = join(self.exp_path,'Images')
        return [join(im_folder,f) for f in sorted(listdir(im_folder)) if f.endswith('.tif')]
    
    @property
    def registered_imgs_lst(self)-> list[PathLike]:
        """Return the list of paths of registered images."""
        im_folder = join(self.exp_path,'Images_Registered')
        return [join(im_folder,f) for f in sorted(listdir(im_folder)) if f.endswith('.tif')]
    
    @property
    def blured_imgs_lst(self)-> list[PathLike]:
        """Return the list of paths of blured images."""
        im_folder = join(self.exp_path,'Images_Blured')
        return [join(im_folder,f) for f in sorted(listdir(im_folder)) if f.endswith('.tif')]

    @property
    def threshold_masks_lst(self)-> list[PathLike]:
        """Return the list of mask paths from the threshold segmentation."""
        mask_folder = join(self.exp_path,'Masks_Threshold')
        return [join(mask_folder,f) for f in sorted(listdir(mask_folder)) if f.endswith('.tif')]
    
    @property
    def cellpose_masks_lst(self)-> list[PathLike]:
        """Return the list of mask paths from the cellpose segmentation."""
        mask_folder = join(self.exp_path,'Masks_Cellpose')
        return [join(mask_folder,f) for f in sorted(listdir(mask_folder)) if f.endswith(('.tif','.npy'))]
    
    @property
    def man_tracked_masks_lst(self)-> list[PathLike]:
        """Return the list of mask paths from the manual tracking."""
        mask_folder = join(self.exp_path,'Masks_Manual_Track')
        return [join(mask_folder,f) for f in sorted(listdir(mask_folder)) if f.endswith('.tif')]
    
    @property
    def gnn_tracked_masks_lst(self)-> list[PathLike]:
        """Return the list of mask paths from the GNN tracking."""
        mask_folder = join(self.exp_path,'Masks_GNN_Track')
        return [join(mask_folder,f) for f in sorted(listdir(mask_folder)) if f.endswith('.tif')]
    
    @property
    def iou_tracked_masks_lst(self)-> list[PathLike]:
        """Return the list of mask paths from the IoU tracking."""
        mask_folder = join(self.exp_path,'Masks_IoU_Track')
        return [join(mask_folder,f) for f in sorted(listdir(mask_folder)) if f.endswith('.tif')]
    
    @property
    def is_analysed(self)-> bool:
        """Check if the experiment has been analysed, by looking for the df_analysis.csv file"""
        if 'df_analysis.csv' in listdir(self.exp_path):
            return True
        return False
        
    def save_df_analysis(self, df_analysis: pd.DataFrame)-> None:
        """Save pandas dataframe as a csv file in the experiment folder."""
        df_analysis.to_csv(join(self.exp_path,'df_analysis.csv'),index=False)
    
    def load_df_analysis(self, data_overwrite: bool=False)-> pd.DataFrame:
        """Load the df_analysis.csv file as a pandas dataframe if it exists. If not return an empty dataframe."""
        if self.is_analysed and not data_overwrite:
            return pd.read_csv(join(self.exp_path,'df_analysis.csv'))
        else:
            return pd.DataFrame()
    
    def save_as_json(self)-> None:
        """Save the experiment settings as a json file in the experiment folder by replacing each class object by its dict."""
        main_dict = self.__dict__.copy()
        main_dict['img_properties'] = self.img_properties.__dict__
        main_dict['analysis'] = self.analysis.__dict__
        main_dict['preprocess'] = self.preprocess.__dict__
        main_dict['segmentation'] = self.segmentation.__dict__
        main_dict['tracking'] = self.tracking.__dict__
        
        with open(join(self.exp_path,'exp_settings.json'),'w') as fp:
            json.dump(main_dict,fp,indent=4)
    
################### Init functions ###################  
def init_from_json(json_path: str)-> Experiment:
    """Function that initialize the Experiment class from a json file. All necessary class arguments would be initialized
    even in nested branches.
    
    Args:
        json_path: str, path to the json file
    Returns:
        Experiment: Experiment class initialized
    Notes:
        The json file is already structured as the different branches of the Experiment class. 
        So here each branch settings are replaced by the corresponding initialized class."""
    with open(json_path) as fp:
        meta = json.load(fp)
    meta['img_properties'] = ImageProperties.from_dict(ImageProperties,meta['img_properties'])
    meta['analysis'] = Analysis.from_dict(Analysis,meta['analysis'])
    meta['preprocess'] = PreProcess.from_dict(PreProcess,meta['preprocess'])
    meta['segmentation'] = Segmentation.from_dict(Segmentation,meta['segmentation'])
    meta['tracking'] = Tracking.from_dict(Tracking,meta['tracking'])
    return Experiment.from_dict(Experiment,meta)
    
def init_from_dict(input_dict: dict)-> Experiment:
    """Function that initialize the Experiment class from a dictionary. All necessary class arguments would be initialized
    even in nested branches.
    
    Args:
        input_dict: dict, dictionary containing the class arguments
    Returns:
        Experiment: Experiment class initialized
    Notes:
        The dictionary is not structured as the different branches of the Experiment class. Each args are in the main dict.
        So the main dict is always given as in input for all the branches."""
    
    input_dict['img_properties'] = ImageProperties.from_dict(ImageProperties,input_dict)
    input_dict['analysis'] = Analysis.from_dict(Analysis,input_dict)
    input_dict['preprocess'] = PreProcess.from_dict(PreProcess,input_dict)
    input_dict['segmentation'] = Segmentation.from_dict(Segmentation,input_dict)
    input_dict['tracking'] = Tracking.from_dict(Tracking,input_dict)
    return Experiment.from_dict(Experiment,input_dict)



if __name__ == '__main__':
    # json_path = '/Users/benhome/BioTool/GitHub/cp_dev/Test_images/Run2/c2z25t23v1_nd2_s1/exp_settings.json'
    
    # settings = Settings.from_json(Settings,json_path=json_path)
    # print(settings)
    # exp = init_from_json(json_path)
    # print(exp.process.background_sub)
    # d = {'a':1,'b':2,'c':3,'d':4,'e':5}
    # proc = Process.from_dict(Process,d)
    # print(type(proc))
    channel_seg = 'RFP'
    # threshold_seg = {channel_seg:{'method':"MANUAL",'threshold':10.4}}
    cellpose_seg = {'RFP':{'model_settings':{'gpu':True,'net_avg':False,'device':None,'diam_mean':30.,'residual_on':True,
                              'style_on':True,'concatenation':False,'nchan':2},'cellpose_eval':{'batch_size':8,'channels':[0,0],'channel_axis':None,'z_axis':None,
            'invert':False,'normalize':True,'diameter':60.,'do_3D':False,'anisotropy':None,
            'net_avg':False,'augment':False,'tile':True,'tile_overlap':0.1,'resample':True,
            'interp':True,'flow_threshold':0.4,'cellprob_threshold':0.0,'min_size':500,
            'stitch_threshold':0.0,'rescale':None,'progress':None,'model_loaded':False}}}
    iou_tracking = {'BFP':{'mask_fold_src':'Masks_Cellpose','stitch_thres_percent':0.75,
                                        'shape_thres_percent':0.1,'n_mask':10}}
    masks = Segmentation(cellpose_seg=cellpose_seg,iou_tracking=iou_tracking)
    
    masks_dict = {}
    for field in fields(masks):
        name = field.name
        value = list(getattr(masks,name).keys())
        if value:
            masks_dict[name] = value
    print(masks_dict)
    if 'iou_tracking' in masks_dict:
        del masks_dict['cellpose_seg']
        
    print(masks_dict)