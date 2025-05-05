from pathlib import Path

from scipy.ndimage import distance_transform_edt
import numpy as np
import pandas as pd

from pipeline.utilities.data_utility import load_stack, get_exp_props


def extract_pixelwise(frame_idx: int, roi_mask_paths: list[Path], img_paths: list[Path], ref_mask_paths: list[Path], channels: list[str] | None = None, resolution: float | None = None) -> pd.DataFrame:
    """
    For a given frame, compute for every pixel inside the ROI-mask:
      - its distance to the nearest ref-mask pixel (in µm if resolution given)
      - its intensity in each channel
    Returns a DataFrame with one row per pixel.
    """
    # 1) load the reference mask & compute distance‐map
    ref = load_stack(ref_mask_paths, frame_range=frame_idx, return_2D=True)
    ref_bool = ref.astype(bool)
    # we want distance OUTSIDE the ref mask, so invert it
    dmap = distance_transform_edt(~ref_bool)
    if resolution:
        dmap = dmap * resolution

    # 2) load your ROI mask
    roi = load_stack(roi_mask_paths, frame_range=frame_idx, return_2D=True)
    # find all the ROI pixels
    ys, xs = np.nonzero(roi)

    # 3) load image channels (if you have multiple)
    img_stacks = load_stack(img_paths, frame_range=frame_idx, return_2D=True)
    
    # 4) for each ROI pixel, grab dmap + intensities
    data = {'dmap': dmap[ys, xs].astype(int)}
    if channels is None:
        channels = get_exp_props(img_paths)[0]
    
    for ch in channels:
        # load the image stack for this channel
        img = img_stacks[ch]
        # grab the intensity at this pixel
        data[ch] = img[ys, xs]

    # 5) assemble DataFrame
    df = pd.DataFrame(data)
    df["frame"] = frame_idx + 1
    return df
