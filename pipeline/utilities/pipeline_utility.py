from __future__ import annotations
from typing import Callable

def is_run_from_ipython()-> bool:
    """Check if the pipeline is run in a notebook or not"""
    from IPython import get_ipython
    return get_ipython() is not None

if is_run_from_ipython():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


def progress_bar(*args,**kwargs)-> Callable:
    """Progress bar function from the tqdm library. Either the notebook or the terminal version is used. See tqdm documentation for more information.
    
    Main Args:
        iterable: iterable object
        desc: str, description of the progress bar
        colour: str, colour of the progress bar
        total: int, total number of iterations, used in multiprocessing"""
    return tqdm(*args,**kwargs)

def pbar_desc(desc: str)-> str:
    """Get the description of the progress bar"""
    if is_run_from_ipython():
        return desc
    return f"\033[94m{desc}\033[0m"
