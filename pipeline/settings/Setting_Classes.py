from __future__ import annotations
from dataclasses import dataclass, field, fields

PREPROCESS_KEYS = ["bg_sub","chan_shift","frame_shift","blur"]
SEGMENTATION_KEYS = ["cellpose","threshold"]
TRACKING_KEYS = ["iou_track", "gnn_track", "man_track"]
ANALYSIS_KEYS = ['extract_data']

@dataclass
class BaseSettings:
    def __post_init__(self)-> None:
        """Check the settings dict and initialize any branch of the class with the same name of the key, if value is True"""
        # Get all the branch names of the class
        branch_name = [f.name for f in fields(self)]
        for k,v in self.settings.items():
            # Skip if key not in the class
            if k not in branch_name:
                continue
            # Initiate branch and unpack settings
            setattr(self, k, v)
        self.update_overwrite()
    
    def update_overwrite(self, overwrite_all: bool=False)-> None:
        """Update the overwrite of all subsequent methods. For example, if bg_sub has overwrite to True, 
        then all other preprocesses will also need to get overwritten since they use bg_sub images"""
        active_branches = self.get_active_branches
        current_overwrite = self.get_current_overwrite
        
        if overwrite_all:
            new_overwrite = [True for _ in range(len(current_overwrite))]
            self.set_new_overwrite(active_branches,new_overwrite)
            return
        
        # Get the new overwrite list, if the previous is true then change the next to true, else keep the same
        new_overwrite = []; is_False = True
        for i in range(len(current_overwrite)):
            if current_overwrite[i] == False and is_False:
                new_overwrite.append(current_overwrite[i])
            elif current_overwrite[i] == True and is_False:
                new_overwrite.append(current_overwrite[i])
                is_False = False
            elif not is_False:
                new_overwrite.append(True)# Update the overwrite attribute
        
        # Update the overwrite attribute
        self.set_new_overwrite(active_branches,new_overwrite)
        return
    
    def set_new_overwrite(self, active_branches: list[str], new_ow: list[bool])-> None:
        """Set the overwrite attribute of the active branches from the new overwrite list"""
        for i,branch in enumerate(active_branches):
            temp_dict = getattr(self,branch)
            temp_dict['overwrite'] = new_ow[i]
            setattr(self,branch,temp_dict)

    @property
    def get_active_branches(self)-> list[str]:
        return [branch.name for branch in fields(self) 
                if hasattr(self,branch.name) and branch.name != 'settings']
    
    @property
    def get_current_overwrite(self)-> list[bool]:
        return [getattr(self,branch)['overwrite'] for branch in self.get_active_branches]

@dataclass
class PreProcessSettings(BaseSettings):
    settings: dict
    bg_sub: dict = field(init=False)
    chan_shift: dict = field(init=False)
    frame_shift: dict = field(init=False)
    blur: dict = field(init=False)
        
@dataclass
class SegmentationSettings(BaseSettings):
    settings: dict
    cellpose: dict = field(init=False)
    threshold: dict = field(init=False)

@dataclass
class TrackingSettings(BaseSettings):
    settings: dict
    iou_track: dict = field(init=False)
    gnn_track: dict = field(init=False)
    man_track: dict = field(init=False)
    
@dataclass
class AnalysisSettings(BaseSettings):
    settings: dict
    extract_data: dict = field(init=False)
    draw_mask: dict = field(init=False)
    
################# main Class #################
@dataclass
class Settings:
    settings: dict
    preprocess: PreProcessSettings = field(init=False)
    segmentation: SegmentationSettings = field(init=False)
    tracking: TrackingSettings = field(init=False)
    analysis: AnalysisSettings = field(init=False)
    overwrite: bool = False
    
    def __post_init__(self)-> None:
        if self.settings['init']["overwrite"]:
            self.overwrite = True
        
        # Unpack the settings
        pre_dict = {k:v[1] for k,v in self.settings.items() if k in PREPROCESS_KEYS and v[0]}
        if pre_dict:
            self.preprocess = PreProcessSettings(pre_dict)
            # If upstream process have overwrite then update the overwrite of the segmentation
            if self.overwrite:
                self.preprocess.update_overwrite(overwrite_all=True)
            # If any of the preprocess has overwrite then set the overwrite to True for downstream applications
            if any(self.preprocess.get_current_overwrite):
                self.overwrite = True
                
        seg_dict = {k:v[1] for k,v in self.settings.items() if k in SEGMENTATION_KEYS and v[0]}
        if seg_dict:
            self.segmentation = SegmentationSettings(seg_dict)
            # If upstream process have overwrite then update the overwrite of the segmentation
            if self.overwrite:
                self.segmentation.update_overwrite(overwrite_all=True)
            # If any of the segmentation has overwrite then set the overwrite to True for downstream applications
            if any(self.segmentation.get_current_overwrite):
                self.overwrite = True
        
        track_dict = {k:v[1] for k,v in self.settings.items() if k in TRACKING_KEYS and v[0]}
        if track_dict:
            self.tracking = TrackingSettings(track_dict)
            # If upstream process have overwrite then update the overwrite of the tracking
            if self.overwrite:
                self.tracking.update_overwrite(overwrite_all=True)
            # If any of the tracking has overwrite then set the overwrite to True for downstream applications
            if any(self.tracking.get_current_overwrite):
                self.overwrite = True
        
        analysis_dict = {k:v[1] for k,v in self.settings.items() if k in ANALYSIS_KEYS and v[0]}
        if analysis_dict:
            self.analysis = AnalysisSettings(analysis_dict)
            # If upstream process have overwrite then update the overwrite of the analysis
            if self.overwrite:
                self.analysis.update_overwrite(overwrite_all=True)
            
        
    