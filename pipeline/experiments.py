from utility import Utility
from os.path import join
from os.path import isdir,exists
from os import mkdir,sep,listdir
from shutil import copy,rmtree
from tifffile import imread, imwrite
import numpy as np
import csv
import pandas as pd
from cellpose import models, core
from cellpose.io import logger_setup

from skimage import draw
from IPython.display import clear_output
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

class Exp_Indiv(Utility):
    def __init__(self,exp_path,channel_seg):
        self.exp_path = exp_path
        
        if exists(join(sep,self.exp_path+sep,'exp_properties.pickle')):
            self.exp_prop = Exp_Indiv.open_exp_prop(exp_path=self.exp_path)
            if 'img_process' not in self.exp_prop: self.exp_prop['img_process'] = {}
        else:
            raise ValueError(f"{self.exp_path} has no preprocessed experiment. Please run the Experiments class first")

        # Extract para
        exp_para = self.exp_prop['metadata']
        self.channel_list = exp_para['channel_list']
        self.channel_seg = channel_seg
        self.x_size = exp_para['x']
        self.y_size = exp_para['y']
        self.z_size = exp_para['z']
        self.frames = exp_para['t']
        self.chan_numb = exp_para['c']

    def get_chanNpath(self,channel_seg,keyword):
        # Create folder to save masks
        masks_path = join(sep,self.exp_path+sep,keyword)
        if not isdir(masks_path):
            mkdir(masks_path)

        # Setting up channel seg
        if channel_seg:
            chan_seg = channel_seg
        else:
            chan_seg = self.channel_seg
        return masks_path,chan_seg
    
    def cellpose_segment(self,imgFold,channel_seg=None,nucMarker=None,seg_ow=False,stitch=None,do_log=True,**kwargs):
        """Function to run cellpose segmentation. See https://github.com/MouseLand/cellpose for more details.
        This function rely on settings defined in the Config class (see Config.cellposeConfig()).

        Args:
            - display (bool): Display or not the output segmentation. 
            - idx_list (int or [int]): Frame index or list of indexes to display. If set to None, choose at most 3 random indexes to display.
            - seg_ow (bool): Overwrite or not existing segmentations.
            - model (dict): Manual settings for the model of Cellpose
        """
        # Get path and channel
        masks_cp_path,chan_seg = self.get_chanNpath(channel_seg=channel_seg,keyword='Masks_CP')
        
        # Check if masks already exist
        if any(chan_seg in file for file in listdir(masks_cp_path)) and not seg_ow:
            print(f"-> Segmented masks already exists for the '{chan_seg}' channel of exp. '{self.exp_path}'\n")
        else:
            # Set log on
            if do_log: logger_setup();
            clear_output()
            print(f"-> Segmenting images for the '{chan_seg}' channel of exp. '{self.exp_path}'")
            
            # Setup Cellpose
            cpMod,cpRun,chan_lst = self.get_cp_settings(nucMarker=nucMarker,chan_seg=chan_seg,stitch=stitch,**kwargs)
            
            for frame in range(self.frames):
                # load image
                imgs = Exp_Indiv.load_stack(imgFold_path=join(sep,self.exp_path+sep,imgFold),
                                                channel_list=chan_lst,
                                                input_range=[frame])
                # Add the channel axis for multichannel images
                if nucMarker:
                    if cpRun['channel_axis']==None: cpRun['channel_axis'] = imgs.ndim-1
                # Run Cellpose. Returns 4 variables
                model = models.CellposeModel(**cpMod)
                masks_cp, __, __, = model.eval(imgs,**cpRun)
                # Save mask
                Exp_Indiv.save_mask(mask_stack=masks_cp,save_dir=masks_cp_path,batch=[frame],z_size=self.z_size,chan_seg=chan_seg,frames=self.frames)
            
            # Save settings
            cp_dict = {'cpModel_settings':cpMod,'cpRun_settings':cpRun}
            self.exp_prop['img_process']['cp_seg'] = cp_dict
            self.exp_prop['fct_inputs']['cellpose_segment'] = {'imgFold':imgFold,'channel_seg':channel_seg,'nucMarker':nucMarker,'stitch':stitch}
            self.exp_prop['fct_inputs']['cellpose_segment'].update(kwargs)
            if 'channel_seg' in self.exp_prop:
                if 'Masks_CP' in self.exp_prop['channel_seg']:
                    if chan_seg not in self.exp_prop['channel_seg']['Masks_CP']:
                        self.exp_prop['channel_seg']['Masks_CP'].append(chan_seg)
                else:
                    self.exp_prop['channel_seg']['Masks_CP'] = [chan_seg]
            else:
                self.exp_prop['channel_seg'] = {'Masks_CP':[chan_seg]}
            Exp_Indiv.save_exp_prop(exp_path=self.exp_path,exp_prop=self.exp_prop)

    def stitch_masks(self,maskFold='Masks_CP',channel_seg=None,stitch_threshold=0.75,shape_threshold=0.2,stitch_ow=False,n_mask=5,): #TODO: remove log from loading mask
        
        if self.frames==1:
            print('Not a time sequence, triming mask will be ignored')
        else:
            # Get path and channel
            masks_stitch_path,chan_seg = self.get_chanNpath(channel_seg=channel_seg,keyword='Masks_stitched')
            
            # Stitch            
            if any(chan_seg in file for file in listdir(masks_stitch_path)) and not stitch_ow:
                print(f'\nExp. {self.exp_path} is already stitched for the "{chan_seg}" channel')
            else:
                clear_output()
                print(f"Stitching masks for {self.exp_path} with '{chan_seg}' channel")
                # Load masks
                maskFold_path = join(sep,self.exp_path+sep,maskFold)
                masks = Exp_Indiv.load_mask(maskFold_path=maskFold_path,channel_seg=chan_seg)

                # Stiching over the whole stack
                masks = Exp_Indiv.modif_stitch3D(masks=masks,stitch_threshold=stitch_threshold)
                
                # Check shape size for detecting merged cells
                new_mask = np.zeros((masks.shape))
                for obj in list(np.unique(masks))[1:]:
                    temp = masks.copy()
                    temp[temp!=obj] = 0
                    t,__,__=np.where(temp!=0)
                    f_lst,size_lst=np.unique(t,return_counts=True)
                    mean_size = np.mean(size_lst)
                    up = mean_size+mean_size*shape_threshold # shape_threshold is the max % of up or down allowed
                    down = mean_size-mean_size*shape_threshold
                    temp[f_lst[np.where((size_lst<down)|(size_lst>up))]] = 0
                    new_mask += temp

                # Re-assign the new value to the masks and obj
                print('--> Reassigning masks value')
                for n, val in enumerate(list(np.unique(new_mask))):
                    new_mask[new_mask == val] = n
                
                # Morph
                print('--> Morphing masks')
                masks = Exp_Indiv.morph(mask_stack=new_mask,n_mask=n_mask)

                # Trim massk
                print('--> Trimming masks')
                for obj in list(np.unique(masks))[1:]:
                    if len(set(np.where(masks==obj)[0]))!=self.frames:
                        masks[masks==obj] = 0

                # Save mask
                print('--> Saving masks')
                Exp_Indiv.save_mask(mask_stack=masks,save_dir=masks_stitch_path,batch=list(range(self.frames)),z_size=self.z_size,chan_seg=chan_seg,frames=self.frames)
                
                # Save exp_prop
                self.exp_prop['fct_inputs']['stitch_masks'] = {'stitch_ow':stitch_ow,'maskFold':maskFold,'channel_seg':channel_seg,'stitch_threshold':stitch_threshold}
                if 'Masks_stitched' in self.exp_prop['channel_seg']:
                    if chan_seg not in self.exp_prop['channel_seg']['Masks_stitched']:
                        self.exp_prop['channel_seg']['Masks_stitched'].append(chan_seg)
                else:
                    self.exp_prop['channel_seg']['Masks_stitched'] = [chan_seg]
                Exp_Indiv.save_exp_prop(exp_path=self.exp_path,exp_prop=self.exp_prop)

    def cell_compart(self,maskFold,channel_seg=None,rad_ero=10,rad_dil=None,compart_ow=False,dil_iteration=1,ero_iteration=1):
        
        # Get path and channel
        mask_compart_path,chan_seg = self.get_chanNpath(channel_seg=channel_seg,keyword='Masks_Compartment')
        
        # Compart
        if any(chan_seg in file for file in listdir(mask_compart_path)) and not compart_ow:
            print(f'Exp. {self.exp_path} is already compartmentalized for the "{channel_seg}" channel\n')
            
        else:
            # Load mask path
            mask_stack = Exp_Indiv.load_mask(maskFold_path=join(sep,self.exp_path+sep,maskFold),channel_seg=chan_seg)
            # Log
            print(f"---> Compartmentalizing loaded masks")
            
            # Dilate?
            if rad_dil:
                mask_stack = Exp_Indiv.dilate_mask(mask_stack=mask_stack,exp_para=self.exp_prop['metadata'],rad_dil=rad_dil,iterations=dil_iteration)

            # Erode mask input
            mask_erode = Exp_Indiv.erode_mask(mask_stack=mask_stack,exp_para=self.exp_prop['metadata'],rad_ero=rad_ero,iterations=ero_iteration)

            # Save mask
            mask_mb = mask_stack-mask_erode
            Exp_Indiv.save_mask(mask_stack=mask_mb,save_dir=mask_compart_path,batch=list(range(self.frames)),z_size=self.z_size,chan_seg=chan_seg,motif='mb',frames=self.frames)
            Exp_Indiv.save_mask(mask_stack=mask_erode,save_dir=mask_compart_path,batch=list(range(self.frames)),z_size=self.z_size,chan_seg=chan_seg,motif='cyto',frames=self.frames)
            Exp_Indiv.save_mask(mask_stack=mask_stack,save_dir=mask_compart_path,batch=list(range(self.frames)),z_size=self.z_size,chan_seg=chan_seg,motif='full',frames=self.frames)            
            
            # Save settings to exp_prop
            self.exp_prop['fct_inputs']['cell_compart'] = {'rad_ero':rad_ero,'rad_dil':rad_dil,'maskFold':maskFold,'channel_seg':channel_seg,'dil_iteration':dil_iteration,'ero_iteration':ero_iteration,'compart_ow':compart_ow}
            if 'Masks_Compartment' in self.exp_prop['channel_seg']:
                if chan_seg not in self.exp_prop['channel_seg']['Masks_Compartment']:
                    self.exp_prop['channel_seg']['Masks_Compartment'].append(chan_seg)
            else:
                self.exp_prop['channel_seg']['Masks_Compartment'] = [chan_seg]
            Exp_Indiv.save_exp_prop(exp_path=self.exp_path,exp_prop=self.exp_prop)
    
    def cell_class(self,maskFold,primary_channel,secondary_channel,rad_ero=10,class_ow=False,**kwargs):
        # Create folder to save the masks
        mask_class_path = join(sep, self.exp_path + sep, 'Masks_Class')
        if not isdir(mask_class_path):
            mkdir(mask_class_path)

        pos_name = f"{primary_channel}_POS_{secondary_channel}"
        neg_name = f"{primary_channel}_NEG_{secondary_channel}"

        # Check if exists
        if any(name in file for file in listdir(mask_class_path) for name in [pos_name,neg_name]) and not class_ow:
            # Log
            print(f'Exp. {self.exp_path} is already classified for the channels {primary_channel} and {secondary_channel}\n')
        else:
            # Load masks
            mask_stack1 = Exp_Indiv.load_mask(maskFold_path=join(sep,self.exp_path+sep,maskFold),channel_seg=primary_channel)
            mask_stack2 = Exp_Indiv.load_mask(maskFold_path=join(sep,self.exp_path+sep,maskFold),channel_seg=secondary_channel)

            # Erode secondary
            mask_stack2 = Exp_Indiv.erode_mask(mask_stack=mask_stack2,exp_para=self.exp_prop['metadata'],rad_ero=rad_ero,**kwargs)

            # Classify cells
            sec_coord = tuple(zip(*np.where(mask_stack2!=0)))
            pos = np.zeros((mask_stack1.shape)).astype('uint16'); neg = np.zeros((mask_stack1.shape)).astype('uint16')
            pos_lst = []; neg_lst = []
            for obj in list(np.unique(mask_stack1))[1:]:
                temp = mask_stack1.copy()
                temp[temp!=obj] = 0
                pri_coord=tuple(zip(*np.where(temp!=0)))

                # Check intersection of both set of coord
                if set(pri_coord)&set(sec_coord):
                    pos += temp
                    pos_lst.append(obj)
                else:
                    neg += temp
                    neg_lst.append(obj)
            
            # Save masks
            Exp_Indiv.save_mask(mask_stack=pos,save_dir=mask_class_path,batch=list(range(self.frames)),z_size=self.z_size,chan_seg=pos_name,frames=self.frames)
            Exp_Indiv.save_mask(mask_stack=neg,save_dir=mask_class_path,batch=list(range(self.frames)),z_size=self.z_size,chan_seg=neg_name,frames=self.frames)

            # Save settings to exp_prop
            self.exp_prop['fct_inputs']['cell_class'] = {'maskFold':maskFold,'primary_channel':primary_channel,'rad_ero':rad_ero,'secondary_channel':secondary_channel,'class_ow':class_ow}
            self.exp_prop['fct_inputs']['cell_class'].update(kwargs)
            self.exp_prop['channel_seg']['Masks_Class'] = (primary_channel,secondary_channel)
            self.exp_prop['masks_process']['classification']= {'folder':maskFold,'pos':pos_lst,'neg':neg_lst}
            Exp_Indiv.save_exp_prop(exp_path=self.exp_path,exp_prop=self.exp_prop)

    def threshold_seg(self,imgFold,thres,blur,channel_seg=None,thres_ow=False,**kwargs): 
        """Function to apply a simple threshold to get mask of the input images. Only apply to 2D images. Images can also be blured before process.

        Args:
            - blur (bool): Whether to apply a gaussian blur or not on input images. Requires additional args, see kwargs.
            - thres (int): Threshold value to create mask. If set to None, then threshold is determined automatically.
            - thres_ow (bool): Overwrite existing mask.
            - kwargs (any, optional): Additional args for gaussian blur if ==True. If nothing or only partially filled, default values will be loaded:
                - blur_kernel (tuple, optional): Size of y and x of the matrix to apply the blur. X and y must be odd intengers greater or equal to 3. Defaults to (55,55).
                - blur_sigma (int, optional): Gaussian kernel standard deviation. Defaults to 100.
                - blur_ow (bool, optional): Overwrite existing blurred images. Defaults to False."""
        # Get path and channel
        mask_st_path,chan_seg = self.get_chanNpath(channel_seg=channel_seg,keyword='Masks_SimpleThreshold')
        
        # Unpack kwargs
        dblur = {'blur_kernel':(15,15),'blur_sigma':5,'blur_ow':False} # Default kwargs for gaussian blur
        if kwargs:
            for k,v in kwargs.items():
                if k in dblur.keys():
                    dblur[k] = v
                else:
                    raise AttributeError(f"kwargs '{k}' is not valid. Only {list(dblur.keys())} are accepted")
        # Overwrite threshold?
        if dblur['blur_ow']==True:
            thres_ow = True
        dblur['imgFold_path'] = join(sep,self.exp_path+sep,imgFold)

        # Check if masks already exist
        if any(chan_seg in file for file in listdir(mask_st_path)) and not thres_ow:
            # Log
            print(f"-> Simple threshold images already exist for '{chan_seg}' channel of exp. '{self.exp_path}'\n")
        else:
            # Blur experiment's image?
            if blur: self.exp_prop,blur_img_path = Exp_Indiv.blur_img(**dblur)

            # Run thresholding
            tempval = []
            for f in range(self.frames):
                # load image
                if blur: im = Exp_Indiv.load_stack(imgFold_path=blur_img_path,channel_list=chan_seg,input_range=[f])
                else:    im = Exp_Indiv.load_stack(imgFold_path=join(sep,self.exp_path+sep,imgFold),channel_list=chan_seg,input_range=[f])
                # Convert to maxIP if necessary
                if self.z_size>1: im = np.amax(im,axis=0); print("Warning, for 3D data, images will be processed as MaxIP and resulting mask will also be MaxIP")
                # Apply threshold
                mask,ret = Exp_Indiv.thresholding(img=im,thres=thres)
                tempval.append(ret)
                # Save mask
                Exp_Indiv.save_mask(mask_stack=mask,save_dir=mask_st_path,batch=[f],frames=1,z_size=1,chan_seg=chan_seg)
            retVal = np.round(np.mean(tempval))

            # log
            if thres:
                print(f"-> Creating simple mask with a threshold of {retVal} for {self.exp_path}")
            else:
                print(f"-> Creating simple mask with an AUTOMATIC threshold of {retVal} for {self.exp_path}")    

            # Save settings
            self.exp_prop['fct_inputs']['exp_thresholding'] = {'imgFold':imgFold,'channel_seg':channel_seg,'thres':retVal,'blur':blur,
                                                                'thres_ow':thres_ow,}
            if blur:
                self.exp_prop['fct_inputs']['exp_thresholding'].update(kwargs)

            if 'channel_seg' in self.exp_prop:
                if 'Masks_SimpleThreshold' in self.exp_prop['channel_seg']:
                    if chan_seg not in self.exp_prop['channel_seg']['Masks_SimpleThreshold']:
                        self.exp_prop['channel_seg']['Masks_SimpleThreshold'].append(chan_seg)
                else:
                    self.exp_prop['channel_seg']['Masks_SimpleThreshold'] = [chan_seg]
            else:
                self.exp_prop['channel_seg'] = {'Masks_SimpleThreshold':[chan_seg]}
            Exp_Indiv.save_exp_prop(exp_path=self.exp_path,exp_prop=self.exp_prop)

    def man_mask(self, channel_seg=None, csv_name='ManualTracking',radius=10, morph = True, n_mask=2, mantrack_ow=False):
        # Get path and channel
        masks_man_path,chan_seg = self.get_chanNpath(channel_seg=channel_seg,keyword='Masks_ManualTrack')
        
        # Run manual tracking           
        if any(chan_seg in file for file in listdir(masks_man_path)) and not mantrack_ow:
            print(f"-> Segmented masks already exists for the '{chan_seg}' channel of exp. '{self.exp_path}'\n")
        else:
            #excpected file name: csv_name + channel_seg .csv
            # all_files = listdir(self.exp_path)    
            # csv_file_name = list(filter(lambda f: csv_name in f and f.endswith('.csv'), all_files))
            csv_file_name = []
            for f in listdir(self.exp_path):
                if f.endswith('.csv'):
                    csv_file_name.append(f)

            print(csv_file_name)
            if chan_seg in str(csv_file_name):
                csv_file_name = [x for x in csv_file_name if chan_seg in x][0]
            else:
                csv_file_name = csv_file_name[0]
            csvpath = join(sep,self.exp_path+sep,csv_file_name)
            data = pd.read_csv(csvpath, encoding= 'unicode_escape', sep=None, engine='python')
            
            #get values needed later in the run from the metadata
            interval = self.exp_prop['metadata']['interval_sec']
            pixel_microns = self.exp_prop['metadata']['pixel_microns']
            
            x_head = [x for x in data.keys() if 'x ' in x][0]
            y_head = [x for x in data.keys() if 'y ' in x][0]
            if self.frames > 1: #check for t > 1, otherwise use PID to iterate later
                t_head = [x for x in data.keys() if 't ' in x][0]
                timecolumn = True
            else:
                t_head = 'PID'
                timecolumn = False
            
            data = data[['TID',x_head,y_head,t_head]] # reduce dataframe to only needed columns
            data = data.dropna() # drop random occuring empty rows from excel/csv
            data = data.astype(float)
            
            if 'micron' in x_head:
                data[x_head] = data[x_head]/pixel_microns #recalculate from microns to pixel
            if 'micron' in y_head:    
                data[y_head] = data[y_head]/pixel_microns
            
            # get framenumber out of timestamp
            if 'min' in t_head:
                interval = interval/60
            elif 'h' in t_head:
                interval = interval/3600
            if timecolumn:
                data[t_head] = round(data[t_head]/interval)
            else:
                data[t_head] = (data[t_head])-1
            data = data.astype(int)
             
            masks_man = np.zeros((self.frames,self.y_size,self.x_size), dtype=int)                    
            for __, row in data.iterrows():
                rr, cc = draw.disk((row[y_head],row[x_head]), radius=radius, shape=masks_man[0].shape)
                if all(masks_man[row[t_head]][rr, cc]!=0): # Check for existing track
                    try: masks_man[row[t_head]][rr+1, cc+1] = row['TID'] # This could fail if 2 objs are exactly on the same corner poisiton (Highly unlikely!!)
                    except: masks_man[row[t_head]][rr-1, cc-1] = row['TID']

                else:
                    masks_man[row[t_head]][rr, cc] = row['TID']
            if morph:
                masks_man = masks_man.astype('uint16') #change to uint16, otherwise other function later will get messed up
                masks_man = Exp_Indiv.morph(mask_stack=masks_man, n_mask=n_mask)

            # Save mask
            for f in range(self.frames):
                f_name = '_f%04d'%(f+1)
                for z in range(self.z_size):
                    z_name = '_z%04d'%(z+1)
                    if self.frames==1:
                        if self.z_size==1:
                            imwrite(join(sep,masks_man_path+sep,f'mask_{chan_seg}.tif'),masks_man[0])
                        else:
                            imwrite(join(sep,masks_man_path+sep,f'mask_{chan_seg}'+z_name+'.tif'),masks_man[0][z,...])
                    else:
                        if self.z_size==1:
                            imwrite(join(sep,masks_man_path+sep,f'mask_{chan_seg}'+f_name+'.tif'),masks_man[f])
                        else:
                            imwrite(join(sep,masks_man_path+sep,f'mask_{chan_seg}'+f_name+z_name+'.tif'),masks_man[f][z,...])
            
            # Save settings
            self.exp_prop['fct_inputs']['man_mask'] = {'channel_seg':channel_seg,'csv_name':csv_name,'radius':radius,'mantrack_ow':mantrack_ow}
            if 'channel_seg' in self.exp_prop:
                if 'Masks_ManualTrack' in self.exp_prop['channel_seg']:
                    if chan_seg not in self.exp_prop['channel_seg']['Masks_ManualTrack']:
                        self.exp_prop['channel_seg']['Masks_ManualTrack'].append(chan_seg)
                else:
                    self.exp_prop['channel_seg']['Masks_ManualTrack'] = [chan_seg]
            else:
                self.exp_prop['channel_seg'] = {'Masks_ManualTrack':[chan_seg]}
            Exp_Indiv.save_exp_prop(exp_path=self.exp_path,exp_prop=self.exp_prop)

    def get_bax_settings(self,imgFold,save_path,**kwargs):
        bax_settings = {'file':[],'minWellR':'NaN','maxWellR':'NaN','channelColors':'1 1 1','channelMin':0,'zStacked':0,'channelMax':1,
                                'use':1,'SegOldVersion':'none','SegSave':1,'SegAlgorithm':'Segment_import','bits':16,'SegImportFolder':'Segmentation',
                                'TrackDeathShift':2.00E-05,'TrackMaxDeathProb':1,'TrackMaxMigScore':0,'TrackMigInOut':0,'TrackNumNeighbours':3,
                                'TrackSingleIdleState':0,'TrackBipartiteMatch':1,'TrackFalsePos':1,'TrackSaveFPAsCells':0,'TrackCentroidOffset':0,
                                'TrackMergeWatersheds':1,'TrackMergeOverlapMaxIter':0,'foiErosion':0,'TrackSaveIterations':0,'TrackSavePTC':0,
                                'TrackSaveCTC':0,'TrackEvaluateCTC':0,'TrackSelectFromGT':0,'condition':'Unspecified','magnification':1,
                                'startT':0,'authorStr':'EMPTY','dT':0,'TrackMergeBrokenMaxArea':0,'TrackMergeBrokenRatio':0.75,'sequenceLength':'EMPTY',
                                'pixelSize':0.65,'numZ':[],'TrackZSpeedStd':3,'TrackMigLogLikeList':'MigLogLikeList_uniformClutter','TrackXSpeedStd':12,
                                'pCnt0':0.2,'pCnt1':0.7,'pCnt2':0.1,'pCntExtrap':0.25,'pSplit':0,'pDeath':0,'TrackPAppear':0.001,'TrackPDisappear':0.001}
        chan_tag = ""
        for i in range(len(self.channel_list)):
            if i==0:
                chan_tag += self.channel_list[i]
            else:
                chan_tag += ':'+self.channel_list[i]
        bax_settings['channelTags'] = chan_tag
        bax_settings['ChannelNames'] = chan_tag.upper()
        bax_settings['file'] = imgFold
        
        # Unpack kwargs
        if kwargs:
            for k,v in kwargs.items():
                if k in bax_settings:
                    bax_settings[k] = v
                else:
                    raise AttributeError(f"kwargs '{k}' is not valid. Only {list(bax_settings.keys())} are accepted")
        
        # 3D?
        bax_settings['numZ'] = self.z_size
        if bax_settings['numZ']==1:
            bax_settings['TrackMigLogLikeList'] = 'MigLogLikeList_uniformClutter'
            bax_settings['TrackZSpeedStd'] = 'EMPTY'
        else:
            bax_settings['TrackMigLogLikeList'] = 'MigLogLikeList_3D'      
        
        # Convert settings values to string and Save settings for baxter
        for k,v in bax_settings.items():
            bax_settings[k] = str(v)
        fieldnames = bax_settings.keys()
        with open(join(sep,save_path+sep,'Setting.csv'), 'w', encoding='UTF8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows([bax_settings])
        f.close()
        return bax_settings, join(sep,save_path+sep,'Setting.csv')             

    def get_cp_settings(self,nucMarker,chan_seg,stitch,**kwargs):
        # Setup model
        builin_models = ['cyto','nuclei','tissuenet','livecell','cyto2','CP','CPx','TN1','TN2','TN3','LC1','LC2','LC3','LC4']
        cpMod = {'gpu':True,'model_type':'cyto2','net_avg':False,'device':None,'pretrained_model':False,'diam_mean':30.,
                'residual_on':True,'style_on':True,'concatenation':False,'nchan':2} # Default kwargs for cellpose model
        cpMod['gpu'] = core.use_gpu()

        # Setup run
        cpRun = {'batch_size':8,'channels':[0,0],'channel_axis':None,'z_axis':None,
                'invert':False,'normalize':True,'diameter':60.,'do_3D':False,'anisotropy':2.0,
                'net_avg':False,'augment':False,'tile':True,'tile_overlap':0.1,'resample':True,
                'interp':True,'flow_threshold':0.4,'cellprob_threshold':0.0,'min_size':500,
                'stitch_threshold':0.0,'rescale':None,'progress':None,'model_loaded':False} # Default kwargs for cellpose run
        if nucMarker:
            cpRun['channels'] = [1,2]
            chan_lst = [chan_seg,nucMarker]
        else:
            chan_lst = [chan_seg]

        if self.z_size>1:
            cpRun['z_axis'] = 0
            if stitch:
                cpRun['stitch_threshold'] = stitch
                cpRun['anisotropy'] = None
            else:
                cpRun['do_3D'] = True
        else:
            cpRun['anisotropy'] = None
        
        # Unpack kwargs
        if kwargs:
            for k,v in kwargs.items():
                if k in cpRun:
                    cpRun[k] = v
                elif k in cpMod:
                    if k=='model_type':
                        if v in builin_models:
                            cpMod[k]=v
                        else:
                            cpMod['model_type']=None
                            cpMod['pretrained_model']=v
                    elif k=='pretrained_model':
                        cpMod['model_type']=None
                        cpMod['pretrained_model']=v
                    else:
                        cpMod[k] = v
                else:
                    raise AttributeError(f"{k} is not part of {list(cpMod.keys())+list(cpRun.keys())}")
        else:
            print("Loading defaults value for Cellpose segmentation...")
        return cpMod,cpRun,chan_lst



################################################################
# Supp function

def test_cellpose(imgFold_path, channel_list = ['red'], channel_seg=None,z_size=1,stitch=None, **kwargs):
    # Create folder to save masks
    masks_cptest_path = join(sep,imgFold_path+sep,'Masks_CP_Test')
    if not isdir(masks_cptest_path):
        mkdir(masks_cptest_path)
        
    # Setting up channel seg
    if channel_seg and channel_list:
        idx_seg = channel_list.index(channel_seg)
    else:
        idx_seg = 0

    foldertiff = len([elem for elem in listdir(imgFold_path) if (elem.endswith('.tiff') or elem.endswith('.tif'))])
    frames = int(foldertiff/len(channel_list)/z_size)

    
    exp_list = []
    for chan in channel_list:
        chan_list = []
        for frame in range(frames):
            f_lst = []
            for im in sorted(listdir(imgFold_path)):
                # To be able to load either _f3digit.tif or _f4digit.tif
                ndigit = len(im.split('_')[1][1:].split('.')[0])
                if im.startswith(chan) and im.__contains__(f'_f%0{ndigit}d'%(frame+1)):
                    f_lst.append(imread(join(sep,imgFold_path+sep,im)))
            chan_list.append(f_lst)
        exp_list.append(chan_list)
    if len(channel_list)==1:
        img_stack = np.squeeze(np.stack(exp_list))
    else:
        img_stack = np.moveaxis(np.squeeze(np.stack(exp_list)), [0], [-1])
        
    
    if frames==1:
        imgs = [img_stack.copy()]
    else:
        imgs = [img_stack[f,...] for f in range(frames)]
        
    # Setup model
    builin_models = ['cyto','nuclei','tissuenet','livecell','cyto2','CP','CPx','TN1','TN2','TN3','LC1','LC2','LC3','LC4']
    cpMod = {'gpu':True,'model_type':'cyto2','net_avg':False,'device':None,'pretrained_model':False,'diam_mean':30.,
            'residual_on':True,'style_on':True,'concatenation':False,'nchan':2} # Default kwargs for cellpose model
    cpMod['gpu'] = core.use_gpu()

    # Setup run
    cpRun = {'batch_size':8,'channels':[idx_seg+1,0],'channel_axis':None,'z_axis':None,
                        'invert':False,'normalize':True,'diameter':30.,'do_3D':False,'anisotropy':2.0,
                        'net_avg':False,'augment':False,'tile':True,'tile_overlap':0.1,'resample':True,
                        'interp':True,'flow_threshold':0.4,'cellprob_threshold':0.0,'min_size':500,
                        'stitch_threshold':0.0,'rescale':None,'progress':None,'model_loaded':False} # Default kwargs for cellpose run

    # Unpack kwargs
    if kwargs:
        for k,v in kwargs.items():
            if k in cpRun:
                cpRun[k] = v
            elif k in cpMod:
                if k=='model_type':
                    if v in builin_models:
                        cpMod[k]=v
                    else:
                        cpMod['model_type']=None
                        cpMod['pretrained_model']=v
                elif k=='pretrained_model':
                    cpMod['model_type']=None
                    cpMod['pretrained_model']=v
                else:
                    cpMod[k] = v
            else:
                raise AttributeError(f"{k} is not part of {list(cpMod.keys())+list(cpRun.keys())}")
    else:
        print("Loading defaults value for Cellpose segmentation...")
    
    # Adjust run settings
    if cpRun['channel_axis']==None:
        cpRun['channel_axis'] = imgs[0].ndim-1
    if z_size>1:
        cpRun['z_axis'] = 0
        if stitch:
            cpRun['stitch_threshold'] = stitch
            cpRun['anisotropy'] = None
        else:
            cpRun['do_3D'] = True
    else:
        cpRun['anisotropy'] = None
    
        
    # Set log on
    logger_setup();
    clear_output()
    
    # Run Cellpose. Returns 4 variables
    model = models.CellposeModel(**cpMod)
    masks_cp, __, __, = model.eval(imgs,**cpRun)

    # Save mask
    for f in range(frames):
        f_name = '_f%04d'%(f+1)
        for z in range(z_size):
            z_name = '_z%04d'%(z+1)
            if frames==1:
                if z_size==1:
                    imwrite(join(sep,masks_cptest_path+sep,f'mask_{channel_seg}.tif'),masks_cp[0])
                else:
                    imwrite(join(sep,masks_cptest_path+sep,f'mask_{channel_seg}'+z_name+'.tif'),masks_cp[0][z,...])
            else:
                if z_size==1:
                    imwrite(join(sep,masks_cptest_path+sep,f'mask_{channel_seg}'+f_name+'.tif'),masks_cp[f])
                else:
                    imwrite(join(sep,masks_cptest_path+sep,f'mask_{channel_seg}'+f_name+z_name+'.tif'),masks_cp[f][z,...])

