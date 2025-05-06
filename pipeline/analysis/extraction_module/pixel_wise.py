from __future__ import annotations
from pathlib import Path

from scipy.ndimage import distance_transform_edt
import numpy as np
import pandas as pd

from pipeline.utilities.data_utility import load_stack, get_exp_props, run_multiprocess


def _extract_pixelwise(frame_idx: int, lookup_masks: dict[int, list[Path]], lookup_imgs: dict[int, list[Path]], lookup_ref_masks: dict[int, list[Path]], channels: list[str], pixel_resolution: float | None = None) -> pd.DataFrame:
    """
    For a given frame, compute for every pixel inside the ROI-mask:
      - its distance to the nearest ref-mask pixel (in µm if resolution given)
      - its intensity in each channel
    Returns a DataFrame with one row per pixel.
    """
    # 1) load the reference mask & compute distance‐map
    ref_mask_paths = lookup_ref_masks[frame_idx]
    ref = load_stack(ref_mask_paths, frame_range=frame_idx, return_2D=True)
    ref_bool = ref.astype(bool)
    # we want distance OUTSIDE the ref mask, so invert it
    dmap = distance_transform_edt(~ref_bool)
    if pixel_resolution:
        dmap = np.round(dmap * pixel_resolution).astype(np.uint16)
    else:
        dmap = dmap.astype(np.uint16)

    # 2) load your ROI mask
    mask_paths = lookup_masks[frame_idx]
    roi = load_stack(mask_paths, frame_range=frame_idx, return_2D=True)
    # find all the ROI pixels
    ys, xs = np.nonzero(roi)

    # 3) load image channels (if you have multiple)
    img_paths = lookup_imgs[frame_idx]
    img_stacks = load_stack(img_paths, channels, frame_idx, return_2D=True)
    # Force 3D shape if only one channel
    if img_stacks.ndim == 2:
        img_stacks = img_stacks[..., np.newaxis]
    # 4) for each ROI pixel, grab dmap + intensities
    data = {'dmap': dmap[ys, xs]}
    
    for ch_idx, chan in enumerate(channels):
        # load the image stack for this channel
        img = img_stacks[..., ch_idx]
        # grab the intensity at this pixel
        data[chan] = img[ys, xs]

    # 5) assemble DataFrame
    df = pd.DataFrame(data)
    df["frame"] = frame_idx + 1
    return df

def _create_lookup_map(file_paths: list[Path], n_frames: int) -> dict[int, list[Path]]:
    """
    Create a lookup map for file paths based on frame indices. It will reduce the number of file paths to only those that are relevant for each frame and avoid overhead in the main function.
    """
    lookup_map = {}
    for frame_idx in range(n_frames):
        tag = f"_f{frame_idx+1:04d}"
        lookup_map[frame_idx] = sorted(path for path in file_paths if tag in path.name)
    return lookup_map

#############################################
############### Main function ###############
#############################################
def extract_pixelwise_data(exp_path: Path, img_paths: list[Path], ref_mask_paths: list[Path], mask_paths: list[Path], pixel_resolution: float | None = None, interval_sec: int = None, overwrite: bool = False) -> pd.DataFrame:
    """
    Extract pixelwise data from the given image paths and mask paths.
    
    Parameters:
        exp_path (Path): Path to the experiment folder.
        img_paths (list[Path]): List of image paths.
        ref_mask_paths (list[Path]): List of reference mask paths.
        mask_paths (list[Path]): List of mask paths.
        pixel_resolution (float | None): Pixel resolution in micrometers. If None, no conversion is applied.
        overwrite (bool): If True, overwrite existing data.

    Returns:
        pd.DataFrame: DataFrame containing the extracted pixelwise data.
    """
    # Check if the data has already been extracted
    parquet_file = exp_path.joinpath("pixelwise_data.parquet")
    if parquet_file.exists() and not overwrite:
        print(f"  --> Loading existing data from \033[94m{parquet_file}\033[0m")
        return pd.read_parquet(parquet_file)
    
    print(f" --> Extracting data from \033[94m{parquet_file}\033[0m")
    # Load the experiment properties
    channels, _, nframes, _ = get_exp_props(img_paths)

    # Process each frame
    fixed_args = {
        'lookup_masks': _create_lookup_map(mask_paths, nframes),
        'lookup_imgs': _create_lookup_map(img_paths, nframes),
        'lookup_ref_masks': _create_lookup_map(ref_mask_paths, nframes),
        'channels': channels,
        'pixel_resolution': pixel_resolution}
    dfs = run_multiprocess(_extract_pixelwise, range(nframes), fixed_args)
    
    # Concatenate all DataFrames
    pixel_df = pd.concat(dfs, ignore_index=True)
    
    # Add the time in seconds if interval is provided
    if interval_sec is not None:
        pixel_df['time_sec'] = (pixel_df['frame'] - 1) * interval_sec
    
    pixel_df.to_parquet(parquet_file, index=False)
    return pixel_df



