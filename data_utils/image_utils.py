from pathlib import Path
from typing import List

SUPPORTED_EXTENSIONS = ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg")


def get_image_files(directory: Path) -> List[Path]:
    """
    Gets all image files with supported extensions from a directory.

    Parameters
    ----------
    directory: Path
        Directory to search for images.

    Returns
    -------
    list of Path
        Sorted list of image file paths.
    """
    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(directory.glob(ext))
        files.extend(directory.glob(ext.upper()))
    return sorted(files)
