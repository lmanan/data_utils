import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import zarr
from numcodecs import Blosc
from skimage.io import imread
from tqdm import tqdm

from data_utils.image_utils import get_image_files

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_COMPRESSOR = Blosc(cname="zstd", clevel=7, shuffle=Blosc.BITSHUFFLE)
DEFAULT_SPATIAL_CHUNKS = {2: (256, 256), 3: (64, 256, 256)}


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
        ("group", "U50"),
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


def _resolve_chunks(
    spatial_shape: Tuple[int, ...], spatial_chunks: Optional[Tuple[int, ...]]
) -> Tuple[int, ...]:
    """Chunk shape (1, 1, *spatial), clipped to the array's spatial extent."""
    if spatial_chunks is None:
        spatial_chunks = DEFAULT_SPATIAL_CHUNKS[len(spatial_shape)]
    if len(spatial_chunks) != len(spatial_shape):
        raise ValueError(
            f"spatial_chunks {spatial_chunks} does not match spatial shape {spatial_shape}"
        )
    return (1, 1, *(min(c, s) for c, s in zip(spatial_chunks, spatial_shape)))


def _build_attrs(
    axis_names: Tuple[str, ...],
    voxel_size_um: Optional[Tuple[float, ...]],
    time_resolution_s: float,
) -> dict:
    """Attributes for a (c, t, [z], y, x) dataset: spatial units in nm, time in s.

    Without a voxel size, ``units`` is omitted rather than guessed: consumers
    (e.g. visualizer.read_axis_metadata) then treat the axes as dimensionless
    voxel/frame indices with resolution 1, which is the truth for such a dataset.
    """
    spatial_axes = axis_names[2:]
    n = len(spatial_axes)

    if voxel_size_um is None:
        return {
            "axis_names": list(axis_names),
            "resolution": [1.0] * (n + 1),
            "offset": [0] * (n + 1),
        }

    if len(voxel_size_um) != n:
        raise ValueError(
            f"voxel_size_um {voxel_size_um} does not match spatial axes {spatial_axes}"
        )
    return {
        "axis_names": list(axis_names),
        "resolution": [time_resolution_s, *(v * 1000.0 for v in voxel_size_um)],
        "offset": [0] * (n + 1),
        "units": ["s", *("nm",) * n],
        "voxel_size_um": list(voxel_size_um),
        "time_resolution_s": time_resolution_s,
    }


def create_zarr(
    container_path: str,
    img_dir_names: List[str],
    group_names: List[str],
    mask_dir_names: Optional[List[str]] = None,
    mapping_csv_file_name: Optional[str] = None,
    as_gray: bool = False,
    voxel_size_um: Optional[Tuple[float, ...]] = None,
    time_resolution_s: float = 1.0,
    spatial_chunks: Optional[Tuple[int, ...]] = None,
    compressor: Optional[Blosc] = DEFAULT_COMPRESSOR,
) -> None:
    """
    Create/update a Zarr with images and optionally relabeled masks.

    Parameters
    ----------
    container_path : str
        Path to the Zarr container to create or update.
    img_dir_names : List[str]
        Directories containing image files, one per group.
    group_names : List[str]
        Names for each group group in the Zarr container.
    mask_dir_names : List[str], optional
        Directories containing mask files, one per group. If None, no masks
        are written and mapping_csv_file_name must also be None.
    mapping_csv_file_name : str, optional
        Path to a space-delimited CSV for relabeling masks. Expected columns:
          group id t [z] y x parent_id original_id
        z is optional. Only (t, original_id) are used for relabeling.
        Must be None when mask_dir_names is None.
    as_gray : bool, optional
        If True, only the first channel of multi-channel images is used.
        Defaults to False.
    voxel_size_um : Tuple[float, ...], optional
        Physical voxel size in micrometers, ordered as the spatial axes
        ((z, y, x) or (y, x)). When given, 'resolution' is written in nm and
        'units' as ["s", "nm", ...]. When None (default), the dataset is left
        dimensionless: resolution 1 per axis and no 'units' attribute.
    time_resolution_s : float, optional
        Seconds between frames. Defaults to 1.0. Only written when
        voxel_size_um is given.
    spatial_chunks : Tuple[int, ...], optional
        Chunk shape along the spatial axes; clipped to the array extent.
        Defaults to (64, 256, 256) in 3D and (256, 256) in 2D.
    compressor : Blosc, optional
        Defaults to Blosc zstd, clevel 7, bitshuffle. Pass None to disable.
    """
    if mask_dir_names is None:
        assert (
            mapping_csv_file_name is None
        ), "mapping_csv_file_name must be None when mask_dir_names is None"
    assert len(img_dir_names) == len(group_names)
    if mask_dir_names is not None:
        assert len(mask_dir_names) == len(group_names)

    container = zarr.open(container_path, mode="a")

    # Load mapping
    mapping_all = (
        _load_mapping(mapping_csv_file_name)
        if mapping_csv_file_name is not None
        else None
    )

    mask_dir_names_iter = (
        mask_dir_names if mask_dir_names is not None else [None] * len(group_names)
    )

    for seq_name, img_dir_str, mask_dir_str in zip(
        group_names, img_dir_names, mask_dir_names_iter
    ):
        image_dir = Path(img_dir_str)

        image_fns = get_image_files(image_dir)

        if mask_dir_str is not None:
            mask_dir = Path(mask_dir_str)
            mask_fns = get_image_files(mask_dir)
            if len(image_fns) != len(mask_fns):
                logger.info(
                    f"Sequence '{seq_name}': #images ({len(image_fns)}) != #masks ({len(mask_fns)})"
                )
                logger.info(f"Using the first {len(mask_fns)} frames.")
                image_fns = image_fns[: len(mask_fns)]
        else:
            mask_fns = None

        num_frames = len(image_fns)
        if num_frames == 0:
            logger.warning(f"Sequence '{seq_name}': no files found, skipping.")
            continue

        # Read first frame to determine shape and dtype
        sample_img = imread(image_fns[0])
        if as_gray and sample_img.ndim == 3 and sample_img.shape[-1] in (1, 3, 4):
            sample_img = sample_img[..., 0]
        sample_img_cyx, spatial_axes = _to_cyx(sample_img)

        num_channels = sample_img_cyx.shape[0]
        spatial_shape = sample_img_cyx.shape[1:]  # (Z, Y, X) or (Y, X)
        img_dtype = sample_img.dtype

        # Build zarr shape: (C, T, [Z], Y, X)
        zarr_shape = (num_channels, num_frames, *spatial_shape)
        axis_names = ("c", "t", *spatial_axes)

        chunks = _resolve_chunks(spatial_shape, spatial_chunks)

        # Create zarr datasets with original dtypes
        seq_grp = container.require_group(seq_name)
        img_ds = seq_grp.create_dataset(
            "img",
            shape=zarr_shape,
            chunks=chunks,
            dtype=img_dtype,
            compressor=compressor,
            overwrite=True,
        )

        if mask_fns is not None:
            mask_zarr_shape = (1, num_frames, *spatial_shape)
            mask_ds = seq_grp.create_dataset(
                "mask",
                shape=mask_zarr_shape,
                chunks=chunks,
                dtype=np.uint32,
                compressor=compressor,
                overwrite=True,
            )
            mapping = mapping_all[mapping_all["group"] == seq_name]
            logger.info("Sequence '%s': using %d mapping rows.", seq_name, len(mapping))
            lookup = _build_lookup(mapping)
        else:
            mask_ds = None
            lookup = None

        # Write frames one by one
        logger.info(f"Processing {num_frames} frames...")
        unique_labels_before = set()
        unique_labels_after = set()

        frame_iter = (
            zip(image_fns, mask_fns)
            if mask_fns is not None
            else ((fn, None) for fn in image_fns)
        )
        for t, (im_fn, ma_fn) in enumerate(
            tqdm(frame_iter, total=num_frames, desc=seq_name)
        ):
            img = imread(im_fn)
            if as_gray and img.ndim == 3 and img.shape[-1] in (1, 3, 4):
                img = img[..., 0]

            # Convert image to (C, [Z], Y, X) format
            img_cyx, _ = _to_cyx(img)
            img_ds[:, t] = img_cyx

            if ma_fn is not None:
                mask = imread(ma_fn).astype(np.uint32)

                # Track unique labels before relabeling
                unique_labels_before.update(np.unique(mask).tolist())

                # Relabel mask
                relabeled_mask = _relabel_block(mask, t, lookup)

                # Track unique labels after relabeling
                unique_labels_after.update(np.unique(relabeled_mask).tolist())

                mask_ds[0, t] = relabeled_mask

        # Set attributes
        attrs = _build_attrs(axis_names, voxel_size_um, time_resolution_s)
        seq_grp["img"].attrs.update(attrs)
        if mask_ds is not None:
            seq_grp["mask"].attrs.update(attrs)
            logger.info(
                "Sequence '%s': relabeled masks written. Unique labels: %d -> %d",
                seq_name,
                len(unique_labels_before),
                len(unique_labels_after),
            )

    logger.info("Created/updated container at %s.", container_path)
