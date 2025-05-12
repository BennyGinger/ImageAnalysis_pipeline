"""
Key-Based Path Converter with OS and Style Awareness

Recursively scans a directory for JSON files. For each JSON file:
  1. Determine its parent directory (`parent_dir`).
  2. Detect current OS builder (`PureWindowsPath` on Windows, else `PurePosixPath`).
  3. Read and optionally convert two fields:
       - `exp_path`: should become exactly `parent_dir`, in OS-native style.
       - `img_properties.img_path`: extract original filename using the correct PurePath
         based on the path's original style, then build under `parent_dir`.
     Conversion only occurs if the stored path differs.
  4. Overwrite the JSON file in-place if any changes were made.

This ensures correct filename placement and OS-appropriate separators, regardless
of whether the original paths in JSON used POSIX or Windows style.

Usage:

    from path_converter import convert_json_paths
    convert_json_paths('/path/to/experiments')
"""
import json
import os
import re
from pathlib import Path, PureWindowsPath, PurePosixPath
from typing import Union

# Regex to detect Windows absolute paths (e.g. "C:/" or "C:\\")
_WIN_RE = re.compile(r'^[A-Za-z]:[\\/]')


def _get_builder():
    """Return the PurePath class matching the current OS."""
    return PureWindowsPath if os.name == 'nt' else PurePosixPath


def _detect_style(s: str) -> str:
    """Return 'windows' or 'posix' if s looks like that style, else 'unknown'."""
    if _WIN_RE.match(s):
        return 'windows'
    if s.startswith('/'):
        return 'posix'
    return 'unknown'


def process_json_file(json_file: Path) -> None:
    """Load a JSON file, correct its two key paths, and save if changed."""
    parent_dir = json_file.parent.resolve()
    builder = _get_builder()

    data = json.loads(json_file.read_text())
    changed = False

    # 1. exp_path
    orig_exp = data.get('exp_path', '')
    new_exp = str(builder(parent_dir))
    if orig_exp != new_exp:
        data['exp_path'] = new_exp
        changed = True

    # 2. img_properties.img_path
    img_props = data.get('img_properties', {})
    orig_img = img_props.get('img_path', '')
    if orig_img:
        # Determine original style and appropriate PurePath
        style = _detect_style(orig_img)
        PathClass = PureWindowsPath if style == 'windows' else PurePosixPath
        filename = PathClass(orig_img).name
        # Build new path under parent_dir in OS-native style
        new_img = str(builder(parent_dir) / filename)
        if orig_img != new_img:
            img_props['img_path'] = new_img
            data['img_properties'] = img_props
            changed = True

    # Save only if modifications occurred
    if changed:
        json_file.write_text(json.dumps(data, indent=2))


def convert_json_paths(search_root: Union[str, Path]) -> None:
    """Scan `search_root` recursively for JSON files and apply corrections."""
    base = Path(search_root).resolve()
    for json_file in base.rglob('*.json'):
        try:
            process_json_file(json_file)
        except Exception:
            continue



if __name__ == '__main__':
    input_path = Path("/home/ben/Docker_mount/Test_images/tiff/Run2")
    
    convert_json_paths(input_path)
