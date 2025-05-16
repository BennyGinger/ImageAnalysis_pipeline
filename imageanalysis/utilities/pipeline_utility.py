from __future__ import annotations
from typing import Iterable, Iterator, TypeVar, Union, Callable, cast
from pathlib import Path
from os import PathLike

from IPython.core.getipython import get_ipython


T = TypeVar("T")
PathType = Union[str, Path, PathLike[str]]

def is_run_from_ipython()-> bool:
    """Check if the pipeline is run in a notebook or not"""
    return get_ipython() is not None

def get_corresponding_tqdm() -> Callable[..., Iterator[object]]:
    """
    Returns either tqdm.notebook.tqdm or the standard tqdm.tqdm,
    both of which accept the same signature and yield Iterator[object].
    """
    if is_run_from_ipython():
        from tqdm.notebook import tqdm as notebook_tqdm
        return notebook_tqdm  # type: ignore[return-value]
    from tqdm import tqdm as standard_tqdm
    return standard_tqdm  # type: ignore[return-value]

def progress_bar(iterable: Iterable[T],*,
                desc: str | None = None,
                colour: str | None = None,
                total: int | None = None,) -> Iterator[T]:
    """
    Progress bar function from the tqdm library. Either the notebook or the terminal version is used. See tqdm documentation for more information.
    
    Main Args:
        iterable: iterable object
        desc: str, description of the progress bar
        colour: str, colour of the progress bar
        total: int, total number of iterations, used in multiprocessing
    """
    tqdm_obj = get_corresponding_tqdm()
    return cast(Iterator[T], tqdm_obj(iterable,
                    desc=desc,
                    colour=colour,
                    total=total))

def pbar_desc(desc: str)-> str:
    """Get the description of the progress bar"""
    if is_run_from_ipython():
        return desc
    return f"\033[94m{desc}\033[0m"

