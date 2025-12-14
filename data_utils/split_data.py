from pathlib import Path
from typing import List

import numpy as np
import zarr
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


def load_image(src_path: Path) -> np.ndarray:
    """
    Loads an image file as a numpy array, preserving original dtype.

    Parameters
    ----------
    src_path: Path
        Source image file path.

    Returns
    -------
    np.ndarray
        Image as numpy array with original dtype.
    """
    img = Image.open(src_path)
    return np.array(img)


def to_czyx(img_array: np.ndarray) -> tuple[np.ndarray, tuple]:
    """
    Convert image array to C [Z] Y X layout.

    Parameters
    ----------
    img_array: np.ndarray
        Image array with shape (Y, X), (Y, X, C), (Z, Y, X), or (Z, Y, X, C).

    Returns
    -------
    tuple(np.ndarray, tuple)
        Array with shape (C, Y, X) or (C, Z, Y, X).
    """
    ndim = img_array.ndim

    if ndim == 2:
        # (Y, X) -> (C=1, Y, X)
        return img_array[np.newaxis, ...], ("c, y, x")
    elif ndim == 3:
        # Could be (Z, Y, X) or (Y, X, C)
        if img_array.shape[-1] in (1, 3, 4):
            # (Y, X, C) -> (C, Y, X)
            return np.moveaxis(img_array, -1, 0), ("c, y, x")
        else:
            # (Z, Y, X) -> (C=1, Z, Y, X)
            return img_array[np.newaxis, ...], ("c, z, y, x")
    elif ndim == 4:
        # (Z, Y, X, C) -> (C, Z, Y, X)
        return np.moveaxis(img_array, -1, 0), ("c, z, y, x")
    else:
        raise ValueError(f"Unsupported array shape: {img_array.shape}")


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

    # Load first image and mask to determine shapes and dtypes
    # Convert to C [Z] Y X layout to get final shape
    sample_image, image_axis_names = to_czyx(load_image(image_paths[0]))
    sample_mask, mask_axis_names = to_czyx(load_image(mask_paths[0]))
    # Shape is (C, [Z,] Y, X) - we'll insert T as second dimension
    image_czyx_shape = sample_image.shape  # (C, [Z,] Y, X)
    mask_czyx_shape = sample_mask.shape
    image_dtype = sample_image.dtype
    mask_dtype = sample_mask.dtype

    print("image dtype:", image_dtype, "mask dtype:", mask_dtype)

    # Create zarr container
    zarr_path = base_path / "data.zarr"
    root = zarr.open(str(zarr_path), mode="w")

    # Create groups and arrays for each split
    split_data_map = {
        "train": train_indices,
        "val": val_indices,
        "test": test_indices,
    }

    for split_name, split_indices in split_data_map.items():
        n_split = len(split_indices)
        if n_split == 0:
            continue

        split_group = root.create_group(split_name)

        # Shape: (C, T, [Z,] Y, X) - insert T after C
        image_shape = (image_czyx_shape[0], n_split) + image_czyx_shape[1:]
        mask_shape = (mask_czyx_shape[0], n_split) + mask_czyx_shape[1:]

        # Chunks: (1, 1, [Z,] Y, X) - chunk size 1 for C and T
        image_chunks = (1, 1) + image_czyx_shape[1:]
        mask_chunks = (1, 1) + mask_czyx_shape[1:]

        images_array = split_group.create_dataset(
            "images",
            shape=image_shape,
            dtype=image_dtype,
            chunks=image_chunks,
        )
        masks_array = split_group.create_dataset(
            "masks",
            shape=mask_shape,
            dtype=mask_dtype,
            chunks=mask_chunks,
        )

        print(f"Saving {n_split} {split_name} images...")
        for i, idx in enumerate(split_indices):
            img, _ = to_czyx(load_image(image_paths[idx]))
            mask, _ = to_czyx(load_image(mask_paths[idx]))
            # Write to position [:, i, ...] (all channels, time index i)
            images_array[:, i, ...] = img
            masks_array[:, i, ...] = mask

        split_group["images"].attrs.update(
            {
                "resolution": (1,) * (len(image_axis_names) - 1),
                "offset": (0,) * (len(image_axis_names) - 1),
                "axis_names": image_axis_names,
            }
        )
        split_group["masks"].attrs.update(
            {
                "resolution": (1,) * (len(mask_axis_names) - 1),
                "offset": (0,) * (len(mask_axis_names) - 1),
                "axis_names": mask_axis_names,
            }
        )

    counts = {"train": n_train, "val": n_val, "test": n_test}
    print(
        f"Split complete: {n_train} train, {n_val} val, {n_test} test "
        f"(total: {n_total}) saved to zarr container at {zarr_path}"
    )

    return counts
