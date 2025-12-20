import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import zarr
from PIL import Image

from data_utils.image_utils import get_image_files

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _to_cyx(img: np.ndarray) -> tuple[np.ndarray, tuple[str, ...]]:
    """
    Convert image to (C, [Z], Y, X) layout.

    Parameters
    ----------
    img : np.ndarray
        Image array with shape (Y, X), (Y, X, C), (Z, Y, X), or (Z, Y, X, C).

    Returns
    -------
    tuple[np.ndarray, tuple[str, ...]]
        Converted array and spatial axis names (without 'c' and 't').
    """
    ndim = img.ndim

    if ndim == 2:
        # (Y, X) -> (C=1, Y, X)
        return img[np.newaxis, ...], ("y", "x")
    elif ndim == 3:
        # Could be (Z, Y, X) or (Y, X, C)
        if img.shape[-1] in (1, 3, 4):
            # (Y, X, C) -> (C, Y, X)
            return np.moveaxis(img, -1, 0), ("y", "x")
        else:
            # (Z, Y, X) -> (C=1, Z, Y, X)
            return img[np.newaxis, ...], ("z", "y", "x")
    elif ndim == 4:
        # (Z, Y, X, C) -> (C, Z, Y, X)
        return np.moveaxis(img, -1, 0), ("z", "y", "x")
    else:
        raise ValueError(f"Unsupported image ndim: {ndim}, shape: {img.shape}")


def _load_mapping(mapping_csv_file_name: str) -> np.ndarray:
    """Load mapping CSV, detecting whether 'z' column exists."""
    with open(mapping_csv_file_name, "r", encoding="utf-8") as f:
        header = f.readline().strip().split()  # space-delimited

    dtype = [
        ("sequence", "U50"),
        ("id", "i8"),
        ("t", "i8"),
        ("y", "f8"),
        ("x", "f8"),
        ("parent_id", "i8"),
        ("original_id", "i8"),
    ]
    if "z" in header:
        dtype.insert(3, ("z", "f8"))

    arr = np.genfromtxt(
        mapping_csv_file_name,
        delimiter=" ",  # <-- use "," here if the file is CSV
        names=True,
        dtype=dtype,
        encoding="utf-8",
    )
    return arr


def _build_lookup(mapping: np.ndarray) -> Dict[Tuple[int, int], int]:
    """Map (t, original_id) -> id from a structured numpy array."""
    lookup: Dict[Tuple[int, int], int] = {}
    for row in mapping:
        lookup[(int(row["t"]), int(row["original_id"]))] = int(row["id"])
    return lookup


def _relabel_block(
    arr: np.ndarray, t: int, lookup: Dict[Tuple[int, int], int]
) -> np.ndarray:
    """Relabel a 2D or 3D mask block at time t using (t, original_id) -> new_id, safely."""
    src = arr  # NEVER mutate or compare against the same array
    dst = src.copy()

    old_vals = np.unique(src)
    old_vals = old_vals[old_vals != 0]

    for old in old_vals:
        new = lookup.get((t, int(old)), int(old))
        if new != old:
            dst[src == old] = new  # compare to src, write to dst

    return dst


def create_zarr(
    container_path: str,
    img_dir_names: List[str],
    mask_dir_names: List[str],
    sequence_names: List[str],
    mapping_csv_file_name: str,
) -> None:
    """
    Create/update a Zarr with images and relabeled masks.

    mapping CSV columns expected (space-delimited):
      sequence id t [z] y x parent_id original_id
    - z is optional
    - Only (t, original_id) are used for relabeling
    """
    assert len(img_dir_names) == len(mask_dir_names) == len(sequence_names)
    container = zarr.open(container_path, mode="a")

    # Load mapping
    mapping_all = _load_mapping(mapping_csv_file_name)

    for seq_name, img_dir_str, mask_dir_str in zip(
        sequence_names, img_dir_names, mask_dir_names
    ):
        image_dir = Path(img_dir_str)
        mask_dir = Path(mask_dir_str)

        image_fns = get_image_files(image_dir)
        mask_fns = get_image_files(mask_dir)
        if len(image_fns) != len(mask_fns):
            logger.info(
                f"Sequence '{seq_name}': #images ({len(image_fns)}) != #masks ({len(mask_fns)})"
            )
            logger.info(f"Using the first {len(mask_fns)} frames.")
            image_fns = image_fns[: len(mask_fns)]

        num_frames = len(image_fns)
        if num_frames == 0:
            logger.warning(f"Sequence '{seq_name}': no files found, skipping.")
            continue

        # Read first frame to determine shape and dtype
        sample_img = np.array(Image.open(image_fns[0]))
        sample_img_cyx, spatial_axes = _to_cyx(sample_img)

        num_channels = sample_img_cyx.shape[0]
        spatial_shape = sample_img_cyx.shape[1:]  # (Z, Y, X) or (Y, X)
        img_dtype = sample_img.dtype

        # Build zarr shape: (C, T, [Z], Y, X)
        zarr_shape = (num_channels, num_frames, *spatial_shape)
        axis_names = ("c", "t", *spatial_axes)

        # Mask shape: single channel
        mask_zarr_shape = (1, num_frames, *spatial_shape)

        # Filter mapping for this sequence
        mapping = mapping_all[mapping_all["sequence"] == seq_name]
        logger.info("Sequence '%s': using %d mapping rows.", seq_name, len(mapping))
        lookup = _build_lookup(mapping)

        # Create zarr datasets with original dtypes
        seq_grp = container.require_group(seq_name)
        img_ds = seq_grp.create_dataset(
            "img",
            shape=zarr_shape,
            chunks=(1, 1, *spatial_shape),
            dtype=img_dtype,
            overwrite=True,
        )
        mask_ds = seq_grp.create_dataset(
            "mask",
            shape=mask_zarr_shape,
            chunks=(1, 1, *spatial_shape),
            dtype=np.uint32,
            overwrite=True,
        )

        # Write frames one by one
        logger.info(f"Processing {num_frames} frames...")
        unique_labels_before = set()
        unique_labels_after = set()

        for t, (im_fn, ma_fn) in enumerate(zip(image_fns, mask_fns)):
            img = np.array(Image.open(im_fn))
            mask = np.array(Image.open(ma_fn)).astype(np.uint32)

            # Convert image to (C, [Z], Y, X) format
            img_cyx, _ = _to_cyx(img)

            # Track unique labels before relabeling
            unique_labels_before.update(np.unique(mask).tolist())

            # Relabel mask
            relabeled_mask = _relabel_block(mask, t, lookup)

            # Track unique labels after relabeling
            unique_labels_after.update(np.unique(relabeled_mask).tolist())

            # Write to zarr
            img_ds[:, t] = img_cyx
            mask_ds[0, t] = relabeled_mask

        # Set attributes
        seq_grp["img"].attrs.update(
            {
                "resolution": (1,) * (len(axis_names) - 1),
                "offset": (0,) * (len(axis_names) - 1),
                "axis_names": axis_names,
            }
        )
        seq_grp["mask"].attrs.update(
            {
                "resolution": (1,) * (len(axis_names) - 1),
                "offset": (0,) * (len(axis_names) - 1),
                "axis_names": axis_names,
            }
        )

        logger.info(
            "Sequence '%s': relabeled masks written. Unique labels: %d -> %d",
            seq_name,
            len(unique_labels_before),
            len(unique_labels_after),
        )

    logger.info("Created/updated container at %s.", container_path)
