import logging
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
from PIL import Image

from data_utils.image_utils import get_image_files

# ----------------- Logging Setup -----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)
# -------------------------------------------------


def create_man_track(
    mask_dir_names: List[str],
    output_file_names: List[str],
    divisions: Optional[List[Optional[Dict[int, int]]]] = None,
) -> None:
    """
    Create manual track files from segmentation masks.

    Produces CSV files with columns: id t_start t_end parent_id

    Parameters
    ----------
    mask_dir_names : List[str]
        List of directories containing segmentation mask files.
        Supported formats: tif, tiff, png, jpg, jpeg.
    output_file_names : List[str]
        List of output file paths for the track CSVs.
        Must have same length as mask_dir_names.
    divisions : List[Dict[int, int] | None] | None
        Optional list of dictionaries mapping child track IDs to parent track IDs.
        If None, all parent_ids are set to 0 (no divisions).
        Must have same length as mask_dir_names if provided.

    """
    if len(mask_dir_names) != len(output_file_names):
        raise ValueError(
            f"mask_dir_names ({len(mask_dir_names)}) and output_file_names "
            f"({len(output_file_names)}) must have the same length"
        )

    if divisions is None:
        divisions = [None] * len(mask_dir_names)

    for idx, (mask_dir, output_file, div) in enumerate(
        zip(mask_dir_names, output_file_names, divisions)
    ):
        logger.info(f"Processing directory {idx + 1}/{len(mask_dir_names)}: {mask_dir}")

        mask_file_names = get_image_files(Path(mask_dir))
        num_frames = len(mask_file_names)

        if num_frames == 0:
            raise ValueError(f"No mask files found in {mask_dir}")

        logger.info(f"Found {num_frames} mask files in {mask_dir}")

        # Track the time range for each label ID
        track_times: Dict[int, List[int]] = defaultdict(list)

        logger.info("Processing masks...")
        for time in range(num_frames):
            mask = np.array(Image.open(mask_file_names[time]))
            unique_labels: Set[int] = set(np.unique(mask))
            unique_labels.discard(0)  # Remove background

            for label in unique_labels:
                track_times[label].append(time)

        # Build track data
        track_data: List[List[int]] = []
        for track_id in sorted(track_times.keys()):
            times = track_times[track_id]
            t_start = min(times)
            t_end = max(times)

            if div is not None and track_id in div:
                parent_id = div[track_id]
            else:
                parent_id = 0

            track_data.append([track_id, t_start, t_end, parent_id])

        logger.info(f"Found {len(track_data)} tracks")

        # Save to file
        logger.info(f"Saving track file to: {output_file}")
        np.savetxt(
            output_file,
            np.array(track_data, dtype=int),
            delimiter=" ",
            header="id t_start t_end parent_id",
            fmt="%i",
        )

    logger.info("Track file creation complete.")


# -------------------- Main CLI ------------------------


def main():
    parser = ArgumentParser(
        description="Create manual track files from segmentation masks"
    )
    parser.add_argument(
        "--mask_dir_names",
        type=str,
        nargs="+",
        required=True,
        help="Directories containing segmentation mask files (tif, png, jpg, etc.)",
    )
    parser.add_argument(
        "--output_file_names",
        type=str,
        nargs="+",
        required=True,
        help="Output file paths for the track CSVs",
    )

    args = parser.parse_args()
    create_man_track(
        mask_dir_names=args.mask_dir_names,
        output_file_names=args.output_file_names,
    )


if __name__ == "__main__":
    main()
