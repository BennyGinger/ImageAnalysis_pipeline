from utility import Utility
from experiments import Exp_Indiv
import pandas as pd
from analysis import Analysis
from os.path import join,exists
from os.path import isdir
from os import sep,walk
import matplotlib.pyplot as plt
from math import ceil

# TODO: implement Docker
# TODO: write all the docstrings
# TODO: make all the notebook template
# TODO: implement parallel processing
# TODO: choose the folder automatically
class Experiments(Utility):
    def __init__(self,parent_folder,channel_list,channel_seg,file_type='.nd2'): # TODO: enable tif file
        """Class that regroups all experiments, for pre-, processing, analysis and ploting

        Args:
            - parent_folder (str): Path of a folder which contains all the image files to be analysied. Image files names will be used as experiment name (rename them if necessary). Image files should be orginized in subfolders depending on their condition. 
            - channel_list ([str]): List of active channel labels (channel that you want to keep active). The order of the channels depends on the order of channel used during acquisition.
            - channel_seg (str): Label of the channel to be processed (for segmentation, registration and so on...), not necessarly the one to be measured.
            - file_type (str): Extention of the files to analyse.
        """
        # Check if path is valid
        if not isdir(parent_folder):
            raise ValueError(f"{parent_folder} is not a correct path. Try a full path")
      
        self.parent_folder = parent_folder
        self.file_type = file_type
       
        # Get the path of all the nd2 files in all subsequent folders/subfolders and exp_dict if available
        self.imgS_path = []
        for root , subs, files in walk(self.parent_folder):
            for f in files:
                if f.endswith(self.file_type):
                    self.imgS_path.append(join(sep,root+sep,f))
        self.imgS_path.sort()
        self.channel_list = channel_list
        
        # Channel check
        if channel_seg not in self.channel_list:
            raise ValueError(f"'{channel_seg}' is not in your channel list: {self.channel_list}")
        else:
            self.channel_seg = channel_seg

    def pre_process_all(self,bg_sub='Auto',imseq_ow=False,true_channel_list=None,reg=False,**kwargs):
        """Convert image files into image sequences, that will be stored separatly into different folders. 
        Background is by defaults automatically substracted, and images can also be registered. Save images into 'Images', 
        and if reg into 'Images_Registered' as well.

        Args:
            - bg_sub (str, optional): Method of background subtraction to use on extracted images. 
            Include None, 'Auto' and 'Manual'. Defaults to 'Auto'.
            - imseq_ow (bool, optional): Overwrite existing image sequences. Defaults to False.
            - true_channel_list ([str], optional): List of ALL channel labels, even the ones to be excluded. 
            Defaults to None.
            - reg (bool, optional): To apply registration on extracted images. Defaults to False.
        
        Kwargs (optional):
            - bg_sub: 
                - sigma (float): Standard deviation for the Gaussian kernel. Defaults to 0.0
                - size (int): Size of the averaging kernel. It should be smaller than the foreground structures. 
                Defaults to 7
            - reg: 
                - reg_mtd (str): Reg mtd to apply. Includes 'translation', 'rigid body' (translation + rotation),
                'scaled rotation' (translation + rotation + scaling), 'affine' (translation + rotation + scaling + shearing) and 
                'bilinear' (non-linear transformation; does not preserve straight lines). Defaults to 'translation'
                - reg_ref (str): Select reference image mtd. Includes 'first', 'previous' and 'mean'. Defaults to 'mean'
                - reg_ow (bool): Overwrite existing regitration. Defaults to False.
                - reg_channel (str): Label of channel to be used as reference. Defaults to channel_seg
        """
        # Unpack all kwargs
        defSMO = {'sigma':0.0,'size':7} # Default kwargs for Auto bgsub methods
        defReg = {'reg_mtd':'translation','reg_ref':'mean','reg_ow':False,'reg_channel':self.channel_seg,'chan_shift':False} # Default kwargs for image registration
        if kwargs: # Replace default val with input args
            for k,v in kwargs.items():
                if k in defReg: defReg[k] = v
                elif k in defSMO: defSMO[k] = v
                else:
                    raise AttributeError(f"kwargs '{k}' is not valid. Only accepted entries are:\n\t- for background substraction: {list(defSMO.keys())}\n\t- for image registration: {list(defReg.keys())}")
        if imseq_ow:
            defReg['reg_ow'] = True
        
        # Log
        print(f"Opening all '{self.file_type}' files:\n")
        
        # Create image seqs
        self.exp_dict = {}
        for img_path in self.imgS_path:
            self.exp_dict.update(self.create_imseq(img_path=img_path,imseq_ow=imseq_ow,
                                                   channel_list=self.channel_list,file_type=self.file_type,
                                                   true_channel_list=true_channel_list))
        
        # Apply reg or bg_sub
        self.exp_folder_path = []
        for k in self.exp_dict.keys():
            if self.exp_dict[k]['status']=='active':
                self.exp_folder_path.append(k)
                # Apply bg?
                if bg_sub == 'Auto':
                    if self.exp_dict[k]['img_preProcess']['bg_sub']!='Auto':
                        self.exp_dict[k] = Experiments.smo_bg_sub(imgFold_path=join(sep,k+sep,'Images'),**defSMO)
                    else: print(f"--> 'Auto' background substraction already applied on {k}")
                elif bg_sub == 'Manual':
                    if self.exp_dict[k]['img_preProcess']['bg_sub']!='Manual':
                        self.exp_dict[k] = Experiments.man_bg_sub(imgFold_path=join(sep,k+sep,'Images'))
                    else: print(f"--> 'Manual' background substraction already applied on {k}")
                # Apply reg?
                if reg: self.exp_dict[k] = Experiments.im_reg(imgFold_path=join(sep,k+sep,'Images'),**defReg)
    
    def remove_exp(self,exp_path):
        """Remove experiment to be further processed and analysed. All existing processed images will NOT be deleted.

        Args:
            exp_path (str): Experiment path to be removed (Folder name should start with 's' followed by a number)
        """
        if type(exp_path)==str:
            exp_path = [exp_path]

        for path in exp_path:
            if not exists(join(sep,path+sep,'REMOVED_EXP.txt')):
                open(join(sep,path+sep,'REMOVED_EXP.txt'),'w')
                self.exp_dict[path]['status'] = 'REMOVED'
            
                if hasattr(self,'exp_folder_path'):
                    self.exp_folder_path.remove(path)
    
    def exp_get_chanNpath(self,channel_seg=None,exp_path=None):
        # Get channel
        if channel_seg:
            chan_seg = channel_seg
        else:
            chan_seg = self.channel_seg

        # Setup paths
        if exp_path:
            # Convert to list
            if type(exp_path)==str:
                exp_path = [exp_path]
            if type(exp_path)!=list:
                raise TypeError(f"Only string or list of strings is allowed")

            # change var
            exp_folder_path = exp_path
        else:
            exp_folder_path = self.exp_folder_path
        return chan_seg, exp_folder_path

    def exp_cp_seg(self,imgFold=None,exp_path=None,channel_seg=None,seg_ow=False,nucMarker=None,stitch=None,do_log=True,**kwargs):
        """Runs Cellpose. For details see https://cellpose.readthedocs.io/en/latest/index.html. Save Masks into 'Masks_CP'.

        Args:
            - imgFold (str, optional): Name of image folder to process. Includes 'Images' (raw images) and 'Images_Registered'. Defaults to 'Images'.
            - exp_path (str or [str], optional): List of all experiment to process. If None, then it will process all. Defaults to None.
            - channel_seg (str, optional): Label of channel to segment. If None, it will use channel_seg. Defaults to None.
            - seg_ow (bool, optional): Overwrite existing segmented mask. Defaults to False.
            - nucMarker (str, optional): Label of nuclear channel to use as secondary channel for segmentation. Defaults to None.
            - stitch (float, optional): 3D only - Replace 3D segmentation with 2D seg of each z-plane and then stitch back together to recreate 3D volumes. 
            'stitch' is the threshold number of minimum overlap between 2 cells to stitched. Defaults to None.
            - do_log (bool, optional): Display log of cellpose. Defaults to True.

        Kwargs (optional): 
            Many different parameters can be changed for the input setting, please check https://cellpose.readthedocs.io/en/latest/api.html for full details.
        The main ones are listed below.
        
            - diameter (float): Average diameter of cell to segment (in pixel). One of the main factor that influence segmentation. Defaults to 30.0.
            - flow_threshold (float): Increase to get more masks. Defaults to 0.4.
            - cellprob_threshold (float): Decrease to get more masks. Between -6 and 6 only. Defaults to 0.0.
            - model_type (str): Full path of custom model. Defaults to None.
        """
        # Get channel and path
        chan_seg, exp_folder_path = self.exp_get_chanNpath(channel_seg=channel_seg,exp_path=exp_path)
            
        # run segmentation
        if not hasattr(self,'exps'):
            self.exps = []

            # Run seg  
            for path in exp_folder_path:
               # Seg
                exp = Exp_Indiv(exp_path=path,channel_seg=self.channel_seg)
                if not imgFold:
                    if 'im_reg' in exp.exp_prop['fct_inputs']: imgFold = 'Images_Registered'
                    else: imgFold = 'Images'
                exp.cellpose_segment(imgFold=imgFold,channel_seg=chan_seg,seg_ow=seg_ow,nucMarker=nucMarker,stitch=stitch,do_log=do_log,**kwargs)
                
                # Update list and exp_dict
                self.exps.append(exp)
                self.exp_dict[path] = exp.exp_prop
        else:
            exp_lst = [exp.exp_path for exp in self.exps]

            # Run seg  
            for path in exp_folder_path:
                if path in exp_lst:
                    # Seg
                    if not imgFold:
                        if 'im_reg' in self.exps[exp_lst.index(path)].exp_prop['fct_inputs']: imgFold = 'Images_Registered'
                        else: imgFold = 'Images'
                    self.exps[exp_lst.index(path)].cellpose_segment(imgFold=imgFold,channel_seg=chan_seg,seg_ow=seg_ow,nucMarker=nucMarker,stitch=stitch,do_log=do_log,**kwargs)
                    
                    # Update exp_dict
                    self.exp_dict[path] = self.exps[exp_lst.index(path)].exp_prop
                else:
                    # Create obj and Seg
                    exp = Exp_Indiv(exp_path=path,channel_seg=self.channel_seg)
                    if not imgFold:
                        if 'im_reg' in exp.exp_prop['fct_inputs']: imgFold = 'Images_Registered'
                        else: imgFold = 'Images'
                    exp.cellpose_segment(imgFold=imgFold,channel_seg=chan_seg,seg_ow=seg_ow,nucMarker=nucMarker,stitch=stitch,do_log=do_log,**kwargs)

                    # Update list and exp_dict
                    self.exps.append(exp)
                    self.exp_dict[path] = exp.exp_prop
        
    def exp_track_cells(self,maskFold='Masks_CP',exp_path=None,channel_seg=None,stitch_threshold=0.75,shape_threshold=0.2,stitch_ow=False,n_mask=5):
        """Track masks of 2D cell lines experiment only.

        Args:
            - maskFold (str, optional): Name of masks folder to process. Defaults to 'Masks_CP'.
            - exp_path (str or [str], optional): List of all experiment to process. If None, then it will process all. Defaults to None.
            - channel_seg (str, optional): Label of channel to segment. If None, it will use channel_seg. Defaults to None.
            - stitch_threshold (float, optional): Minimum threshold of overlap between 2 cells to be stitched together. Defaults to 0.75.
            - shape_threshold (float, optional): Allowed percentage for a cell to change shape/size within the track (detect merge/split cells). Defaults to 0.2.
            - n_mask (int, optional): Minimum appearance of mask with the track to be accepted as cell (detect false positive cells, e.g. floating cells). Defaults to 5.
            - stitch_ow (bool, optional): Overwrite existing tracked masks. Defaults to False.
        """
        # Get channel and path
        chan_seg, exp_folder_path = self.exp_get_chanNpath(channel_seg=channel_seg,exp_path=exp_path)
        
        # List of paths
        exp_lst = [exp.exp_path for exp in self.exps]
        
        # Run stitch
        for path in exp_folder_path:
            # Stitch
            self.exps[exp_lst.index(path)].stitch_masks(stitch_threshold=stitch_threshold,channel_seg=chan_seg,stitch_ow=stitch_ow,maskFold=maskFold,shape_threshold=shape_threshold,n_mask=n_mask)

            # Update exp_dict
            self.exp_dict[path] = self.exps[exp_lst.index(path)].exp_prop
    
    def exp_compart(self,maskFold,channel_seg=None,exp_path=None,rad_ero=10,rad_dil=None,compart_ow=False,dil_iteration=1,ero_iteration=1):
        # Get channel and path
        chan_seg, exp_folder_path = self.exp_get_chanNpath(channel_seg=channel_seg,exp_path=exp_path)
        
        # List of paths
        exp_lst = [exp.exp_path for exp in self.exps]

        # Run track  
        for path in exp_folder_path:
            self.exps[exp_lst.index(path)].cell_compart(maskFold=maskFold,channel_seg=chan_seg,rad_ero=rad_ero,rad_dil=rad_dil,compart_ow=compart_ow,dil_iteration=dil_iteration,ero_iteration=ero_iteration)
            
            # Update exp_dict
            self.exp_dict[path] = self.exps[exp_lst.index(path)].exp_prop
    
    def exp_class(self,maskFold,primary_channel,secondary_channel,exp_path=None,rad_ero=10,class_ow=False,**kwargs):
        # Get channel and path
        __,exp_folder_path = self.exp_get_chanNpath(exp_path=exp_path)
        
        # List of paths
        exp_lst = [exp.exp_path for exp in self.exps]

        # Run track  
        for path in exp_folder_path:
            if secondary_channel not in self.exps[exp_lst.index(path)].exp_prop['channel_seg'][maskFold]:
                raise AttributeError(f"The secondary channel {secondary_channel} also need to be process. Please run segmentation+tracking")
                
            self.exps[exp_lst.index(path)].cell_class(maskFold=maskFold,primary_channel=primary_channel,secondary_channel=secondary_channel,rad_ero=rad_ero,class_ow=class_ow,**kwargs)
            # Update exp_dict
            self.exp_dict[path] = self.exps[exp_lst.index(path)].exp_prop

    def exp_analysis(self,imgFold,maskFold,channel_seg=None,exp_path=None,df_ow=False,do_cell_dist=False,maskLabel='wound',ref_mask_ow=False,signal_type='linear',**kwargs): # TODO: manual tag
        # Get channel and path
        chan_seg, exp_folder_path = self.exp_get_chanNpath(channel_seg=channel_seg,exp_path=exp_path)
        
        # Unpack kwargs
        d_Ana = {'interval':None,'tag':None, 'tag_ow':False} 
        d_ExData = {'stim_time':None,'start_baseline':0,'posCont_time':None} #TODO: extract stim time from exp name
        for k,v in kwargs.items():
            if k in d_Ana:
                d_Ana[k] = v
            elif k in d_ExData:
                d_ExData[k] = v

        # Run analysis
        if not hasattr(self,'exps_analysis'):
            self.exps_analysis = []

            # Extract data
            self.masterdf_analysis = pd.DataFrame()
            for path in exp_folder_path:
                anal_exp = Analysis(exp_path=path,channel_seg=self.channel_seg,**d_Ana)
                anal_exp.extract_channelData(imgFold=imgFold,maskFold=maskFold,channel_seg=chan_seg,df_ow=df_ow,signal_type=signal_type,**d_ExData)
                anal_exp.extract_centroids(maskFold=maskFold,channel_seg=chan_seg,df_ow=df_ow)
                if do_cell_dist:
                    # Run ref_mask
                    anal_exp.cell_distance(imgFold=imgFold,df_ow=df_ow,maskLabel=maskLabel,ref_mask_ow=ref_mask_ow)
                # Update exp_prop and concat masterdf
                self.exp_dict[path] = anal_exp.exp_prop
                self.masterdf_analysis = pd.concat([self.masterdf_analysis,anal_exp.df_analysis])
                self.exps_analysis.append(anal_exp)
        else:
            exp_lst = [exp.exp_path for exp in self.exps_analysis]

            # Extract data
            self.masterdf_analysis = pd.DataFrame()
            for path in exp_folder_path:
                if path in exp_lst:
                    self.exps_analysis[exp_lst.index(path)].extract_channelData(imgFold=imgFold,maskFold=maskFold,channel_seg=chan_seg,df_ow=df_ow,**d_ExData)
                    self.exps_analysis[exp_lst.index(path)].extract_centroids(maskFold=maskFold,channel_seg=chan_seg,df_ow=df_ow)
                    if do_cell_dist:
                        self.exps_analysis[exp_lst.index(path)].cell_distance(imgFold=imgFold,df_ow=df_ow,maskLabel=maskLabel)
                    # Update exp_prop and concat masterdf
                    self.exp_dict[path] = self.exps_analysis[exp_lst.index(path)].exp_prop
                    self.masterdf_analysis = pd.concat([self.masterdf_analysis,self.exps_analysis[exp_lst.index(path)].df_analysis])
                else:
                    anal_exp = Analysis(exp_path=path,channel_seg=self.channel_seg,**d_Ana)
                    anal_exp.extract_channelData(imgFold=imgFold,maskFold=maskFold,channel_seg=chan_seg,df_ow=df_ow,**d_ExData)
                    anal_exp.extract_centroids(maskFold=maskFold,channel_seg=chan_seg,df_ow=df_ow)
                    if do_cell_dist:
                        # Run ref_mask
                        anal_exp.cell_distance(imgFold=imgFold,df_ow=df_ow,maskLabel=maskLabel, ref_mask_ow=ref_mask_ow)
                    # Update exp_prop and concat masterdf
                    self.exp_dict[path] = anal_exp.exp_prop
                    self.masterdf_analysis = pd.concat([self.masterdf_analysis,anal_exp.df_analysis])
                    self.exps_analysis.append(anal_exp)
        
        # Save df as csv to parent folder
        self.masterdf_analysis.to_csv(join(sep,self.parent_folder+sep,'masterdf_analysis.csv'),index=False)

    def exp_man_mask(self,exp_path=None,channel_seg=None,csv_name='ManualTracking',radius=10,morph=True,n_mask=2,mantrack_ow=False):
        # Get channel and path
        chan_seg, exp_folder_path = self.exp_get_chanNpath(channel_seg=channel_seg,exp_path=exp_path)
        
        # Run man_mask
        if not hasattr(self,'exps'):
            self.exps = []
            for path in exp_folder_path:
                # Seg
                exp = Exp_Indiv(exp_path=path,channel_seg=self.channel_seg)
                exp.man_mask(channel_seg=chan_seg, csv_name=csv_name, radius=radius,morph=morph, n_mask=n_mask ,mantrack_ow=mantrack_ow)
                # Update exp_dict
                self.exps.append(exp)
                self.exp_dict[path] = exp.exp_prop
        else:
            # List of paths
            exp_lst = [exp.exp_path for exp in self.exps]
        
            # Run stitch
            for path in exp_folder_path:
                if path in exp_lst:
                    # Stitch
                    self.exps[exp_lst.index(path)].man_mask(channel_seg=chan_seg, csv_name=csv_name, radius=radius, morph=morph, n_mask=n_mask, mantrack_ow=mantrack_ow)

                    # Update exp_dict
                    self.exp_dict[path] = self.exps[exp_lst.index(path)].exp_prop
                else:
                    # Seg
                    exp = Exp_Indiv(exp_path=path,channel_seg=self.channel_seg)
                    exp.man_mask(channel_seg=chan_seg, csv_name=csv_name, radius=radius, mantrack_ow=mantrack_ow)
                    # Update exp_dict
                    self.exps.append(exp)
                    self.exp_dict[path] = exp.exp_prop

    def exp_threshold_seg(self,imgFold='Images',do_blur=True,man_threshold=None,channel_seg=None,exp_path=None,thres_ow=False,**kwargs):
        # Get channel and path
        chan_seg, exp_folder_path = self.exp_get_chanNpath(channel_seg=channel_seg,exp_path=exp_path)

        # run segmentation
        if not hasattr(self,'exps'):
            self.exps = []

            # Run seg  
            for path in exp_folder_path:
                # Seg
                exp = Exp_Indiv(exp_path=path,channel_seg=self.channel_seg)
                exp.threshold_seg(imgFold=imgFold,
                                thres=man_threshold,
                                blur=do_blur,
                                channel_seg=chan_seg,
                                thres_ow=thres_ow,
                                **kwargs)
                # Update list and exp_dict
                self.exps.append(exp)
                self.exp_dict[path] = exp.exp_prop
        else:
            exp_lst = [exp.exp_path for exp in self.exps]

            # Run seg  
            for path in exp_folder_path:
                if path in exp_lst:
                    # Seg
                    self.exps[exp_lst.index(path)].threshold_seg(imgFold=imgFold,
                                                                thres=man_threshold,
                                                                blur=do_blur,
                                                                channel_seg=chan_seg,
                                                                thres_ow=thres_ow,
                                                                **kwargs)
                    # Update exp_dict
                    self.exp_dict[path] = self.exps[exp_lst.index(path)].exp_prop
                else:
                    # Create obj and Seg
                    exp = Exp_Indiv(exp_path=path,channel_seg=self.channel_seg)
                    exp.threshold_seg(imgFold=imgFold,
                                    thres=man_threshold,
                                    blur=do_blur,
                                    channel_seg=chan_seg,
                                    thres_ow=thres_ow,
                                    **kwargs)
                    # Update list and exp_dict
                    self.exps.append(exp)
                    self.exp_dict[path] = exp.exp_prop

    def exp_pixel_distance(self,imgFold,maskFold,exp_path=None,channel_seg=None,maskLabel='wound',do_cond_df=False,pix_ana_ow=False,ref_mask_ow=False,interval=None,man_tag=None): # [ ]: modify function to load masterdf if any
        # Get channel and path
        chan_seg, exp_folder_path = self.exp_get_chanNpath(channel_seg=channel_seg,exp_path=exp_path)

        # run pixel_distance
        if exists(join(sep,self.parent_folder+sep,'masterdf_pixel.csv')) and not pix_ana_ow:
            # Load masterdf
            self.masterdf_pixel = pd.read_csv(join(sep,self.parent_folder+sep,'masterdf_pixel.csv'))
            # Recall all experiment
            self.exps_analysis = []
            for path in exp_folder_path:
                # Seg
                exp = Analysis(exp_path=path,channel_seg=self.channel_seg)
                # Add man interval or tag
                if interval: exp.interval = interval
                if man_tag: # input==dict with k=tag anf v=path or [path]
                    if path in man_tag: exp.tag = man_tag[path]
                exp.pixel_distance(imgFold=imgFold,
                                channel_seg=chan_seg,
                                maskFold=maskFold,
                                maskLabel=maskLabel,
                                pix_ana_ow=pix_ana_ow,
                                ref_mask_ow=ref_mask_ow,
                                )
                # Update list and exp_dict
                self.exps_analysis.append(exp)     
        else:
            # Extract data
            self.masterdf_pixel = pd.DataFrame()
            self.exps_analysis = []
            for path in exp_folder_path:
                # Seg
                exp = Analysis(exp_path=path,channel_seg=self.channel_seg)
                # Add man interval or tag
                if interval: exp.interval = interval
                if man_tag: # input==dict with k=tag anf v=path or [path]
                    if path in man_tag: exp.tag = man_tag[path]
                exp.pixel_distance(imgFold=imgFold,
                                channel_seg=chan_seg,
                                maskFold=maskFold,
                                maskLabel=maskLabel,
                                pix_ana_ow=pix_ana_ow,
                                ref_mask_ow=ref_mask_ow,
                                )
                # Update list and exp_dict
                self.exps_analysis.append(exp)
                self.exp_dict[path] = exp.exp_prop
                self.masterdf_pixel = pd.concat([self.masterdf_pixel,exp.df_pixel])
            # Save masterdf
            self.masterdf_pixel.to_csv(join(sep,self.parent_folder+sep,'masterdf_pixel.csv'),index=False)

        # Create conddf
        if do_cond_df:
            for tag in self.masterdf_pixel['tag'].unique():
                # Extract df
                conddf_pixel = self.masterdf_pixel.loc[self.masterdf_pixel['tag']==tag,:].copy()
                # Save conddf
                conddf_pixel.to_csv(join(sep,self.parent_folder+sep,str(tag)+sep,'conddf_pixel.csv'),index=False)

    def pre_plotHM(self,exp_path,maxdt=None):
        # Get channel and path
        __, exp_folder_path = self.exp_get_chanNpath(exp_path=exp_path)
        
        # upper limit?
        if maxdt:
            if isinstance(maxdt,float) or type(maxdt)==int: maxdmap = maxdt
            elif isinstance(maxdt,bool): 
                max_lst = [pd.read_csv(join(sep,path+sep,'df_pixel.csv')).dmap.max() for path in exp_folder_path]
                maxdmap = min(max_lst)
        else: maxdmap = None
        return exp_folder_path,maxdmap
   
    def exp_plot_indHM(self,col_name,deltaF,maxdt=None,intBin=5,col_lim=[0,2],exp_path=None,savedir=None,row_col=None,figsize=None,cbar_label=r'$\Delta$F/F$_{min}$',**kwargs): #TODO: save figure
        # Get attribute
        exp_folder_path,maxdmap = self.pre_plotHM(exp_path=exp_path,maxdt=maxdt)
        
        # Create the subplot
        if row_col: nrow,ncol = row_col
        else:
            if len(exp_folder_path)<4: ncol = len(exp_folder_path)
            else: ncol = 4
            nrow = ceil(len(exp_folder_path)/ncol)
            ncol = ceil(len(exp_folder_path)/nrow) # Adjust col
            if nrow*ncol<len(exp_folder_path): nrow += 1
        
        # plot HM
        r = 0; c = 0 # Initialise the axes
        for path in exp_folder_path:
            if savedir: savepath = savedir
            else: savepath = path
            # Create tag and exp name
            split_path = path.split(sep)[-1].split('_')
            exp_name = '_'.join([path.split(sep)[-2],split_path[0],split_path[-1]])
            # Load df
            df = pd.read_csv(join(sep,path+sep,'df_pixel.csv'))
            # Bin it and plot it
            if maxdt: bin_df = Experiments.pixel_bin(df_pixel=df.loc[df['dmap']<=maxdmap,:].copy(),intBin=intBin,col_name=col_name,deltaF=deltaF)
            else: bin_df = Experiments.pixel_bin(df_pixel=df,intBin=intBin,col_name=col_name,deltaF=deltaF)
            if nrow==1 and ncol==1: Experiments.plot_HM(bin_df,title=exp_name,savedir=savepath,figsize=figsize,cbar_label=cbar_label,col_lim=col_lim,**kwargs)
            else: Experiments.plot_HM(bin_df,title=exp_name,savedir=savepath,figsize=figsize,cbar_label=cbar_label,col_lim=col_lim,**kwargs)
            # Adjust the axes
            c += 1
            if c==ncol: c = 0; r+=1
    
    def exp_plot_condHM(self,col_name,deltaF,maxdt=None,intBin=5,col_lim=[0,2],exp_path=None,savedir=None,row_col=None,figsize=None,cbar_label=r'$\Delta$F/F$_{min}$',**kwargs):
        # Get attribute
        __,maxdmap = self.pre_plotHM(exp_path=exp_path,maxdt=maxdt)
        if not hasattr(self,'masterdf_pixel'):
            self.masterdf_pixel = pd.read_csv(join(sep,self.parent_folder+sep,'masterdf_pixel.csv'))
        # Get tags
        tag_lst = [tag for tag in self.masterdf_pixel['tag'].unique()]

        # Create the subplot
        if row_col: nrow,ncol = row_col
        else:
            if len(tag_lst)<4: ncol = len(tag_lst)
            else: ncol = 4
            nrow = ceil(len(tag_lst)/ncol)
            ncol = ceil(len(tag_lst)/nrow) # Adjust col
            if nrow*ncol<len(tag_lst): nrow += 1

        # plot HM
        r = 0; c = 0 # Initialise the axes
        for tag in self.masterdf_pixel['tag'].unique():
            # Load df
            fig,ax = plt.subplots(1,1,sharey=True,figsize=figsize)
            df = self.masterdf_pixel.loc[self.masterdf_pixel['tag']==tag,:].copy()
            # Bin it and plot it
            if maxdt: bin_df = Experiments.pixel_bin(df_pixel=df.loc[df['dmap']<=maxdmap,:].copy(),intBin=intBin,col_name=col_name,deltaF=deltaF)
            else: bin_df = Experiments.pixel_bin(df_pixel=df,intBin=intBin,col_name=col_name,deltaF=deltaF)
            if nrow==1 and ncol==1: Experiments.plot_HM(bin_df,title=tag,axes=ax,savedir=savedir,cbar_label=cbar_label,col_lim=col_lim,**kwargs)
            else: Experiments.plot_HM(bin_df,title=tag,axes=ax,savedir=savedir,cbar_label=cbar_label,col_lim=col_lim,**kwargs)
            # Adjust the axes
            c += 1
            if c==ncol: c = 0; r+=1


    # TODO: def add_exp_para_var()
    
    # TODO: def read_exp_para()
