from __future__ import annotations
from dataclasses import dataclass,fields,field
import json
from os import sep,listdir,getcwd
from os.path import join
import pandas as pd

@dataclass
class LoadClass:
    def from_dict(self, input_dict: dict)->dict:
        fieldSet = {f.name for f in fields(self) if f.init}
        filteredArgDict = {k : v for k, v in input_dict.items() if k in fieldSet}
        return self(**filteredArgDict)

@dataclass
class Process(LoadClass):
    background_sub: list = field(default_factory=list)
    channel_shift_corrected: list = field(default_factory=list)
    img_registered: list = field(default_factory=list)
    img_blured: list = field(default_factory=list)
    
@dataclass
class Masks(LoadClass):
    threshold_seg: dict = field(default_factory=dict)
    cellpose_seg: dict = field(default_factory=dict)
    iou_tracking: dict = field(default_factory=dict)


@dataclass
class ImageProperties(LoadClass):
    """Get metadata from nd2 or tif file, using ND2Reader or TiffFile and ImageJ"""
    img_width: int
    img_length: int
    n_frames: int
    full_n_channels: int
    n_slices: int
    n_series: int
    img_path: str
    
@dataclass
class Analysis(LoadClass):
    pixel_microns: float
    interval_sec: float
    file_type: str
    level_0_tag: str
    level_1_tag: str
    df_analysis: bool = False

@dataclass
class Experiment(LoadClass):
    exp_path: str
    status: str = 'Not processed'
    active_channel_list: list = field(default_factory=list)
    full_channel_list: list = field(default_factory=list)
    img_properties: ImageProperties = field(default_factory=ImageProperties)
    analysis: Analysis = field(default_factory=Analysis)
    process: Process = field(default_factory=Process)
    masks: Masks = field(default_factory=Masks)

    def __post_init__(self)-> None:
        if 'REMOVED_EXP.txt' in listdir(self.exp_path):
            self.status = 'Removed'
        
        if 'exp_setting.json' in listdir(self.exp_path):
            self = init_from_json(join(sep,self.exp_path+sep,'exp_settings.json'))
    
    @property
    def processed_images_list(self)-> list:
        im_folder = join(sep,self.exp_path+sep,'Images')
        return [join(sep,im_folder+sep,f) for f in sorted(listdir(im_folder)) if f.endswith('.tif')]
    
    @property
    def register_images_list(self)-> list:
        im_folder = join(sep,self.exp_path+sep,'Images_Registered')
        return [join(sep,im_folder+sep,f) for f in sorted(listdir(im_folder)) if f.endswith('.tif')]
    
    @property
    def blur_images_list(self)-> list:
        im_folder = join(sep,self.exp_path+sep,'Images_Blured')
        return [join(sep,im_folder+sep,f) for f in sorted(listdir(im_folder)) if f.endswith('.tif')]

    @property
    def mask_threshold_list(self)-> list:
        mask_folder = join(sep,self.exp_path+sep,'Masks_Threshold')
        return [join(sep,mask_folder+sep,f) for f in sorted(listdir(mask_folder)) if f.endswith('.tif')]
    
    @property
    def mask_cellpose_list(self)-> list:
        mask_folder = join(sep,self.exp_path+sep,'Masks_Cellpose')
        return [join(sep,mask_folder+sep,f) for f in sorted(listdir(mask_folder)) if f.endswith(('.tif','.npy'))]
    
    @property
    def mask_iou_track_list(self)-> list:
        mask_folder = join(sep,self.exp_path+sep,'Masks_IoU_Track')
        return [join(sep,mask_folder+sep,f) for f in sorted(listdir(mask_folder)) if f.endswith('.tif')]
    
    @property
    def time_seq(self)-> list:
        return [round(i*self.analysis.interval_sec,ndigits=2) for i in range(self.img_properties.n_frames)]
    
    def save_df_analysis(self, df_analysis: pd.DataFrame)-> None:
        self.analysis.df_analysis = True
        df_analysis.to_csv(join(sep,self.exp_path+sep,'df_analysis.csv'),index=False)
    
    def load_df_analysis(self, data_overwrite: bool=False)-> pd.DataFrame:
        if self.analysis.df_analysis and not data_overwrite:
            return pd.read_csv(join(sep,self.exp_path+sep,'df_analysis.csv'))
        else:
            return pd.DataFrame()
    
    def save_as_json(self)-> None: #FIXME: Need to save all the path as relative path, or both. as if work from a dev container, the path will be different
        main_dict = self.__dict__.copy()
        main_dict['img_properties'] = self.img_properties.__dict__
        main_dict['analysis'] = self.analysis.__dict__
        main_dict['process'] = self.process.__dict__
        main_dict['masks'] = self.masks.__dict__
        
        with open(join(sep,self.exp_path+sep,'exp_settings.json'),'w') as fp:
            json.dump(main_dict,fp,indent=4)
    
    def set_attribute(self, attr: str, value: any)-> None:
        for field in fields(self):
            if field.name == attr:
                setattr(self,attr,value)
                return
            elif field.name == 'img_properties':
                for img_field in fields(self.img_properties):
                    if img_field.name == attr:
                        setattr(self.img_properties,attr,value)
                        return
            elif field.name == 'analysis':
                for analysis_field in fields(self.analysis):
                    if analysis_field.name == attr:
                        setattr(self.analysis,attr,value)
                        return
            elif field.name == 'process':
                for process_field in fields(self.process):
                    if process_field.name == attr:
                        setattr(self.process,attr,value)
                        return
            elif field.name == 'masks':
                for masks_field in fields(self.masks):
                    if masks_field.name == attr:
                        setattr(self.masks,attr,value)
                        return
    
def init_from_json(json_path: str)-> Experiment:
    with open(json_path) as fp:
        meta = json.load(fp)
    meta['img_properties'] = ImageProperties.from_dict(ImageProperties,meta['img_properties'])
    meta['analysis'] = Analysis.from_dict(Analysis,meta['analysis'])
    meta['process'] = Process.from_dict(Process,meta['process'])
    meta['masks'] = Masks.from_dict(Masks,meta['masks'])
    return Experiment.from_dict(Experiment,meta)
    
def init_from_dict(input_dict: dict)-> Experiment:
    input_dict['img_properties'] = ImageProperties.from_dict(ImageProperties,input_dict)
    input_dict['analysis'] = Analysis.from_dict(Analysis,input_dict)
    input_dict['process'] = Process.from_dict(Process,input_dict)
    input_dict['masks'] = Masks.from_dict(Masks,input_dict)
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
    masks = Masks(cellpose_seg=cellpose_seg,iou_tracking=iou_tracking)
    
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
                
    
    