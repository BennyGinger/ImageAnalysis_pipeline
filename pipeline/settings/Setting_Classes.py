from __future__ import annotations
from dataclasses import dataclass, field, fields


@dataclass
class PreProcessSettings:
    settings: dict
    bg_sub: dict = field(init=False)
    chan_shift: dict = field(init=False)
    register: dict = field(init=False)
    blur: dict = field(init=False)
    
    def __post_init__(self)-> None:
        if self.settings['run_bg_sub']:
            self.bg_sub = self.settings['bg_sub']
        if self.settings['run_chan_shift']:
            self.chan_shift = self.settings['chan_shift']
        if self.settings['run_register']:
            self.register = self.settings['register']
        if self.settings['run_blur']:
            self.blur = self.settings['blur']
        self.update_overwrite()
        
    def update_overwrite(self, overwrite_all: bool=False)-> None:
        active_branches = [f.name for f in fields(self) if hasattr(self,f.name) and f.name != 'settings']
        current_overwrite = [getattr(self,f)['overwrite'] for f in active_branches]

        if overwrite_all:
            new_overwrite = [True for i in range(len(current_overwrite))]
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
        for i,branch in enumerate(active_branches):
            temp_dict = getattr(self,branch)
            temp_dict['overwrite'] = new_ow[i]
            setattr(self,branch,temp_dict)


################# Class ideas #################

# @dataclass
# class BgSub:
#     sigma: float = field(default_factory=float)
#     size: int = field(default_factory=int)
#     overwrite: bool = field(default_factory=bool)
    
# @dataclass
# class ChanShift:
#     reg_channel: str = field(default_factory=str)
#     reg_mtd: str = field(default_factory=str)
#     overwrite: bool = field(default_factory=bool)
    
# @dataclass
# class Register:
#     reg_channel: str = field(default_factory=str)
#     reg_mtd: str = field(default_factory=str)
#     reg_ref: str = field(default_factory=str)
#     overwrite: bool = field(default_factory=bool)
    
# @dataclass
# class Blur:
#     kernel: tuple[int] = field(default_factory=tuple)
#     sigma: int = field(default_factory=int)
#     img_fold_src: str = field(default_factory=str)
#     overwrite: bool = field(default_factory=bool)


# @dataclass
# class Settings:
#     settings: dict
#     bg_sub: dict = field(init=False)
#     chan_shift: dict = field(init=False)
#     register: dict = field(init=False)
#     blur: dict = field(init=False)
    
#     def __post_init__(self)-> None:
#         if self.settings['run_bg_sub']:
#             self.bg_sub = self.settings['bg_sub']
#         if self.settings['run_chan_shift']:
#             self.chan_shift = self.settings['chan_shift']
#         if self.settings['run_register']:
#             self.register = self.settings['register']
#         if self.settings['run_blur']:
#             self.blur = self.settings['blur']
#         self.update_overwrite()
        
#     def update_overwrite(self)-> None:
#         active_branches = [f.name for f in fields(self) if hasattr(self,f.name) and f.name != 'settings']
#         current_overwrite = [getattr(self,f)['overwrite'] for f in active_branches]

#         # Get the new overwrite list, if the previous is true then change the next to true, else keep the same
#         new_overwrite = []; is_False = True
#         for i in range(len(current_overwrite)):
#             if current_overwrite[i] == False and is_False:
#                 new_overwrite.append(current_overwrite[i])
#             elif current_overwrite[i] == True and is_False:
#                 new_overwrite.append(current_overwrite[i])
#                 is_False = False
#             elif not is_False:
#                 new_overwrite.append(True)# Update the overwrite attribute
        
#         # Update the overwrite attribute
#         for i,branch in enumerate(active_branches):
#             temp_dict = getattr(self,branch)
#             temp_dict['overwrite'] = new_overwrite[i]
#             setattr(self,branch,temp_dict)