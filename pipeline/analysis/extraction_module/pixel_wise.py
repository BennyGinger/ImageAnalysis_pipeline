from pathlib import Path

from scipy.ndimage import distance_transform_edt
import numpy as np
import pandas as pd

from pipeline.utilities.data_utility import load_stack


def extract_pixelwise(frame_idx: int, roi_mask_paths: list[Path], img_paths: list[Path], ref_mask_paths: list[Path], channels: list[str] | None = None, resolution: float | None = None) -> pd.DataFrame:
    """
    For a given frame, compute for every pixel inside the ROI-mask:
      - its distance to the nearest ref-mask pixel (in µm if resolution given)
      - its intensity in each channel
    Returns a DataFrame with one row per pixel.
    """
    # 1) load the reference mask & compute distance‐map
    ref = load_stack(ref_mask_paths, frame_range=frame_idx, return_2D=True)
    # we want distance OUTSIDE the ref mask, so invert it
    dmap = distance_transform_edt(ref == 0)
    if resolution:
        dmap = dmap * resolution

    # 2) load your ROI mask
    roi = load_stack(roi_mask_paths, frame_range=frame_idx, return_2D=True)
    # find all the ROI pixels
    ys, xs = np.nonzero(roi)

    # 3) load image channels (if you have multiple)
    #    load_stack can take channel names if your load_stack supports it…
    if channels is None:
        # fallback: assume every plane in img_paths is one channel
        img_stacks = [load_stack([p], frame_range=frame_idx, return_2D=True) 
                      for p in img_paths]
        channels = [f"chan_{i}" for i in range(len(img_stacks))]
    else:
        img_stacks = [load_stack(img_paths, channels=[ch], frame_range=frame_idx, return_2D=True)
                      for ch in channels]

    # 4) for each ROI pixel, grab dmap + intensities
    data = {
        "y": ys,
        "x": xs,
        "distance_to_ref": dmap[ys, xs].astype(float),
    }
    for name, stack in zip(channels, img_stacks):
        data[name] = stack[ys, xs].astype(float)

    # 5) assemble DataFrame
    df = pd.DataFrame(data)
    df["frame"] = frame_idx + 1
    return df
