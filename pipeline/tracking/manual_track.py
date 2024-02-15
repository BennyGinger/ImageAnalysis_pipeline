from __future__ import annotations
from os import listdir, mkdir, sep
from os.path import isdir, join
import numpy as np
import pandas as pd
from skimage import draw
from mask_transformation.mask_morph import morph_missing_mask
from Experiment_Classes import Experiment
from tifffile import imwrite

# FIXME: The input should be a list[Experiment] see cp_segmentation.py and Experiment_Classes.py for more info 
def man_mask(exp_path: str, channel_seg=None, csv_name='ManualTracking',radius=10, morph = True, n_mask=2, mantrack_ow=False):
        # Get path and channel
        # TODO: might need to change that function
        masks_man_path,chan_seg = get_chanNpath(channel_seg=channel_seg,keyword='Masks_ManualTrack')
        
        # Run manual tracking           
        if any(chan_seg in file for file in listdir(masks_man_path)) and not mantrack_ow:
            print(f"-> Segmented masks already exists for the '{chan_seg}' channel of exp. '{exp_path}'\n")
        else:
            #excpected file name: csv_name + channel_seg .csv
            # all_files = listdir(self.exp_path)    
            # csv_file_name = list(filter(lambda f: csv_name in f and f.endswith('.csv'), all_files))
            csv_file_name = []
            for f in listdir(exp_path):
                if f.endswith('.csv'):
                    csv_file_name.append(f)

            print(csv_file_name)
            if chan_seg in str(csv_file_name):
                csv_file_name = [x for x in csv_file_name if chan_seg in x][0]
            else:
                csv_file_name = csv_file_name[0]
            csvpath = join(sep,exp_path+sep,csv_file_name)
            data = pd.read_csv(csvpath, encoding= 'unicode_escape', sep=None, engine='python')
            
            #get values needed later in the run from the metadata
            
            interval = self.exp_prop['metadata']['interval_sec'] # FIXME: This should be a method in the Experiment class
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
             
            masks_man = np.zeros((self.frames,self.y_size,self.x_size), dtype=int)    # FIXME: same as above                
            for __, row in data.iterrows():
                rr, cc = draw.disk((row[y_head],row[x_head]), radius=radius, shape=masks_man[0].shape)
                if all(masks_man[row[t_head]][rr, cc]!=0): # Check for existing track
                    try: masks_man[row[t_head]][rr+1, cc+1] = row['TID'] # This could fail if 2 objs are exactly on the same corner poisiton (Highly unlikely!!)
                    except: masks_man[row[t_head]][rr-1, cc-1] = row['TID']

                else:
                    masks_man[row[t_head]][rr, cc] = row['TID']
            if morph:
                masks_man = masks_man.astype('uint16') #change to uint16, otherwise other function later will get messed up
                
                # FIXME: use the imported morph_missing_mask function to fill in missing masks. see mask_morph.py for more info
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
            
            # FIXME: Save settings as I do in the cp_segmentation.py
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


def get_chanNpath(channel_seg,keyword):
    # Create folder to save masks
    masks_path = join(sep,exp_path+sep,keyword)
    if not isdir(masks_path):
        mkdir(masks_path)

    # Setting up channel seg
    if channel_seg:
        chan_seg = channel_seg
    else:
        chan_seg = self.channel_seg
    return masks_path,chan_seg