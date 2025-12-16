from pathlib import Path
from typing import List

import numpy as np

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


def filter_matching_files(
    image_paths: List[Path], mask_paths: List[Path]
) -> tuple[List[Path], List[Path]]:
    """
    Filters image and mask paths to only include files with matching names
    (by stem, ignoring extension) in both directories.

    Parameters
    ----------
    image_paths: list of Path
        List of image file paths.
    mask_paths: list of Path
        List of mask file paths.

    Returns
    -------
    tuple of (list of Path, list of Path)
        Filtered and sorted lists of matching image and mask paths.
    """
    image_stems = {p.stem: p for p in image_paths}
    mask_stems = {p.stem: p for p in mask_paths}

    common_stems = set(image_stems.keys()) & set(mask_stems.keys())

    if len(common_stems) < len(image_paths) or len(common_stems) < len(mask_paths):
        print(
            f"Warning: Found {len(image_paths)} images and {len(mask_paths)} masks. "
            f"Keeping {len(common_stems)} matching pairs."
        )

    filtered_images = sorted([image_stems[stem] for stem in common_stems])
    filtered_masks = sorted([mask_stems[stem] for stem in common_stems])

    return filtered_images, filtered_masks


def split_indices(
    data_dir: str,
    project_name: str,
    test_fraction: float = 0.15,
    val_fraction: float = 0.15,
    seed: int = 1000,
    consecutive: bool = False,
):
    """
    Computes train, validation, and test indices for image/mask pairs.

    Scans the project directory for matching image/mask pairs, then generates
    indices to split them. First reserves `test_fraction` of total data for
    testing, then reserves `val_fraction` of the remaining data for validation.

    Parameters
    ----------
    data_dir: str
        Path where the project lives.
    project_name: str
        Name of the sub-folder under `data_dir`.
    test_fraction: float
        Fraction of total data to reserve for testing.
    val_fraction: float
        Fraction of remaining (non-test) data to reserve for validation.
    seed: int
        Random seed for reproducible splits.
    consecutive: bool
        If True, takes contiguous blocks of frames instead of random selection.
        Order (sorted by filename): test, then val, then train.

    Returns
    -------
    dict
        Dictionary with keys 'train', 'val', 'test', each mapping to an array of indices.
    """
    base_path = Path(data_dir) / project_name

    # Load source images and masks
    image_dir = base_path / "images"
    mask_dir = base_path / "masks"

    image_paths, mask_paths = filter_matching_files(
        get_image_files(image_dir),
        get_image_files(mask_dir),
    )

    n_total = len(image_paths)
    if n_total == 0:
        raise ValueError(f"No matching image/mask pairs found in {image_dir}")

    indices = np.arange(n_total)

    # Calculate split sizes (use round to avoid 0 allocations for small datasets)
    n_test = round(test_fraction * n_total)
    n_trainval = n_total - n_test
    n_val = round(val_fraction * n_trainval)
    n_train = n_trainval - n_val

    print(
        f"Split sizes: {n_train} train, {n_val} val, {n_test} test (total: {n_total})"
    )

    if consecutive:
        # Consecutive blocks: test first, then val, then train
        test_indices = indices[:n_test]
        val_indices = indices[n_test : n_test + n_val]
        train_indices = indices[n_test + n_val :]
    else:
        # Random split
        np.random.seed(seed)
        np.random.shuffle(indices)
        test_indices = indices[:n_test]
        val_indices = indices[n_test : n_test + n_val]
        train_indices = indices[n_test + n_val :]

    # Create groups and arrays for each split
    split_data_map = {
        "train": train_indices,
        "val": val_indices,
        "test": test_indices,
    }

    return split_data_map
