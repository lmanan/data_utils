from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

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


def load_and_save_as_numpy(src_path: Path, dst_dir: Path) -> Path:
    """
    Loads an image file and saves it as a numpy array.

    Parameters
    ----------
    src_path: Path
        Source image file path.
    dst_dir: Path
        Destination directory to save the numpy array.

    Returns
    -------
    Path
        Path to the saved numpy file.
    """
    img = Image.open(src_path)
    arr = np.array(img)
    dst_path = dst_dir / (src_path.stem + ".npy")
    np.save(dst_path, arr)
    return dst_path


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


def split_data(
    data_dir: str,
    project_name: str,
    test_fraction: float = 0.15,
    val_fraction: float = 0.15,
    seed: int = 1000,
    consecutive: bool = False,
):
    """
    Splits data into train, validation, and test sets.

    First splits all data into (train+val) and test by `test_fraction`.
    Then splits (train+val) into train and val by `val_fraction`.

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
        Dictionary with counts: {'train': n, 'val': n, 'test': n}
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

    print(f"Split sizes: {n_train} train, {n_val} val, {n_test} test (total: {n_total})")

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

    # Create output directories
    splits = ["train", "val", "test"]
    for split in splits:
        for subdir in ["images", "masks"]:
            path = base_path / split / subdir
            if not path.exists():
                path.mkdir(parents=True)
                print(f"Created directory: {path}")

    # Save train set
    print(f"Saving {len(train_indices)} train images...")
    for idx in train_indices:
        load_and_save_as_numpy(image_paths[idx], base_path / "train" / "images")
        load_and_save_as_numpy(mask_paths[idx], base_path / "train" / "masks")

    # Save val set
    print(f"Saving {len(val_indices)} val images...")
    for idx in val_indices:
        load_and_save_as_numpy(image_paths[idx], base_path / "val" / "images")
        load_and_save_as_numpy(mask_paths[idx], base_path / "val" / "masks")

    # Save test set
    print(f"Saving {len(test_indices)} test images...")
    for idx in test_indices:
        load_and_save_as_numpy(image_paths[idx], base_path / "test" / "images")
        load_and_save_as_numpy(mask_paths[idx], base_path / "test" / "masks")

    counts = {"train": n_train, "val": n_val, "test": n_test}
    print(
        f"Split complete: {n_train} train, {n_val} val, {n_test} test "
        f"(total: {n_total}) saved as numpy arrays to {base_path}"
    )

    return counts
