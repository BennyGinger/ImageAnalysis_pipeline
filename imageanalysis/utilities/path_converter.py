"""
Key-Based Path Converter with OS and Style Awareness and Existence Checks

Recursively scans a directory for JSON files. For each JSON file:
  1. Determine its parent directory (`parent_dir`).
  2. Detect current OS builder (`PureWindowsPath` on Windows, else `PurePosixPath`).
  3. Read and convert two fields:
       - `exp_path`: becomes exactly `parent_dir` in OS-native style.
       - `img_properties.img_path`: extract original filename using the correct PurePath
         based on the path's original style, then build under `parent_dir`.
         Verify existence of the constructed image path; raise if missing.
  4. Overwrite the JSON file in-place if any changes were made.

Removes any try/except around JSON processing so that any FileNotFoundError or
ValueError bubbles up to the caller for immediate feedback.

Usage:

    from path_converter import convert_json_paths
    convert_json_paths('/path/to/experiments')
"""
import json
import os
import re
from pathlib import Path, PureWindowsPath, PurePosixPath, PurePath
from typing import Union

# Regex to detect Windows absolute paths (e.g. "C:/" or "C:\\")
_WIN_RE = re.compile(r'^[A-Za-z]:[\\/]')


def _get_builder() -> PurePath:
    """
    Return the PurePath class matching the current OS.
    """
    return PureWindowsPath if os.name == 'nt' else PurePosixPath


def _detect_style(s: str) -> str:
    """
    Return 'windows' or 'posix' if s looks like that style, else 'unknown'.
    """
    if _WIN_RE.match(s):
        return 'windows'
    if s.startswith('/'):
        return 'posix'
    return 'unknown'


def process_json_file(json_file: Path) -> None:
    """
    Load a JSON file, correct its two key paths exp_path and img_properties.img_path, then save it.
    The exp_path is set to the parent directory of the JSON file, and img_properties.img_path
    is set to the filename of the image file in the same directory.
    If the image file does not exist at the new location, a FileNotFoundError is raised.
    The JSON file is modified in place only if changes are made.
    Raises:
        FileNotFoundError: If the image file does not exist at the new location.
        ValueError: If the img_path is invalid and cannot extract the filename.
    """
    folder_dir = json_file.parent.resolve()
    main_dir = folder_dir.parent
    builder = _get_builder()

    # Read JSON file
    data = json.loads(json_file.read_text())
    changed = False

    # Convert exp_path
    orig_exp = data.get('exp_path', '')
    new_exp = str(builder(folder_dir))
    if orig_exp != new_exp:
        data['exp_path'] = new_exp
        changed = True

    # Convert img_properties.img_path
    img_props = data.get('img_properties', {})
    orig_img = img_props.get('img_path', '')
    if orig_img:
        # Determine original style and appropriate PurePath
        style = _detect_style(orig_img)
        PathClass = PureWindowsPath if style == 'windows' else PurePosixPath
        filename = PathClass(orig_img).name
        if not filename:
            raise ValueError(f"Invalid img_path, cannot extract filename: {orig_img}")

        # Build new path under parent_dir in OS-native style
        new_img = str(builder(main_dir).joinpath(filename))

        # Existence check
        if not Path(new_img).exists():
            raise FileNotFoundError(f"Image file not found at expected location: {new_img}")

        if orig_img != new_img:
            img_props['img_path'] = new_img
            data['img_properties'] = img_props
            changed = True

    # Save only if modifications occurred
    if changed:
        json_file.write_text(json.dumps(data, indent=2))


def convert_json_paths(search_root: Union[str, Path]) -> None:
    """
    Scan 'search_root' recursively for JSON files and apply corrections, if needed.
    Any errors (FileNotFoundError/ValueError) will propagate to the caller.
    """
    base = Path(search_root).resolve()
    for json_file in base.rglob('*.json'):
        process_json_file(json_file)



if __name__ == '__main__':
    input_path = Path("/home/ben/Docker_mount/Test_images/tiff/Run2")
    
    convert_json_paths(input_path)
