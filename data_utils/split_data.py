import shutil
from pathlib import Path
from typing import List

import numpy as np

from data_utils.image_utils import get_image_files


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
    remove_source: bool = True,
):
    """
    Splits image/mask pairs into train, validation, and test sets.

    Reads images and masks from `{data_dir}/{project_name}/images` and
    `{data_dir}/{project_name}/masks`, then copies them to
    `train/`, `val/`, and `test/` subdirectories, preserving original format.

    First reserves `test_fraction` of total data for testing, then reserves
    `val_fraction` of the remaining data for validation.

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
    remove_source: bool
        If True, removes the source `images/` and `masks/` directories after splitting.

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

    # Create output directories
    splits = {
        "train": train_indices,
        "val": val_indices,
        "test": test_indices,
    }

    for split_name, split_indices in splits.items():
        split_image_dir = base_path / split_name / "images"
        split_mask_dir = base_path / split_name / "masks"
        split_image_dir.mkdir(parents=True, exist_ok=True)
        split_mask_dir.mkdir(parents=True, exist_ok=True)

        for idx in split_indices:
            img_path = image_paths[idx]
            mask_path = mask_paths[idx]

            # Copy files preserving original format
            shutil.copy2(img_path, split_image_dir / img_path.name)
            shutil.copy2(mask_path, split_mask_dir / mask_path.name)

    if remove_source:
        shutil.rmtree(image_dir)
        shutil.rmtree(mask_dir)

    return {"train": n_train, "val": n_val, "test": n_test}
