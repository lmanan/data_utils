#!/usr/bin/env python3
"""
Extract cropped frames from videos and save to a zarr container.

--timestamps is required and must be provided in the same order as --video.
Cameras are aligned via PTP wall-clock timestamps so that zarr frame t is the
same absolute moment in every camera group. Source fps is derived automatically
from cam_frame_time in the timestamp files.

Usage:
  # 1. Preview crop region on a sample frame (saves annotated PNG):
  python extract_frames.py --preview --crop-x 0 --crop-y 680 --crop-w 2204 --crop-h 980 --video cam_1.mp4

  # 2. Extract frames from all cameras into zarr with groups cam_1/img, cam_2/img, etc.:
  python extract_frames.py \\
      --crop-x 0 --crop-y 680 --crop-w 2204 --crop-h 980 \\
      --video cam_1.mp4 cam_2.mp4 cam_3.mp4 cam_4.mp4 \\
      --timestamps EMERGENT_2002626_*_timestamp.json \\
                   EMERGENT_2002627_*_timestamp.json \\
                   EMERGENT_2002629_*_timestamp.json \\
                   EMERGENT_2002630_*_timestamp.json

  # 3. Same but downsampled to 5 fps:
  python extract_frames.py \\
      --crop-x 0 --crop-y 680 --crop-w 2204 --crop-h 980 \\
      --target-fps 5 \\
      --video cam_1.mp4 cam_2.mp4 cam_3.mp4 cam_4.mp4 \\
      --timestamps EMERGENT_2002626_*_timestamp.json \\
                   EMERGENT_2002627_*_timestamp.json \\
                   EMERGENT_2002629_*_timestamp.json \\
                   EMERGENT_2002630_*_timestamp.json
"""

import argparse
import json
import logging
import math
import subprocess
from pathlib import Path

import numpy as np
import zarr
from skimage.io import imsave
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

DATASET_NAME = "img"
PTP_DATASET_NAME = "ptp_time_ns"


def load_timestamps(json_path: str) -> tuple[np.ndarray | None, float | None]:
    """Parse a camera timestamp JSON and return PTP times and estimated fps.

    The JSON contains one entry per video frame with fields:
      - cam_frame_num:     camera's internal (wrapping) frame counter
      - counter_frame_num: per-camera sequential counter (starts at 1 for each
                           camera independently — NOT a shared hardware counter)
      - cam_frame_time:    camera local clock in nanoseconds
      - ptp_frame_time:    PTP wall-clock time in nanoseconds (sync reference)

    Returns:
        ptp_times:  int64 array of ptp_frame_time values, one per video frame,
                    or None if the field is absent.
        fps:        frame rate estimated from cam_frame_time inter-frame intervals,
                    or None if cam_frame_time is absent.
    """
    data = json.loads(Path(json_path).read_text())

    ptp_times = None
    if data and "ptp_frame_time" in data[0]:
        ptp_times = np.array(
            [int(row["ptp_frame_time"]) for row in data], dtype=np.int64
        )

    fps = None
    if data and "cam_frame_time" in data[0]:
        cam_times = np.array(
            [int(row["cam_frame_time"]) for row in data], dtype=np.int64
        )
        fps = 1e9 / float(np.median(np.diff(cam_times)))

    return ptp_times, fps


def _find_stable_ptp_start(
    ptp_times: np.ndarray, frame_period_ns: float, window: int = 5, tol: float = 0.3
) -> int:
    """Return the index of the first frame with a stable (locked) PTP timestamp.

    EMERGENT cameras' PTP clock typically takes ~80-100 frames to lock after
    recording starts.  During the settling period the inter-frame PTP diffs are
    much smaller than expected (the clock is still catching up), making ptp[0]
    an unreliable reference for sync.

    Stability is defined as `window` consecutive inter-frame diffs that are all
    within `tol` (fraction) of the expected frame period.

    Returns 0 if no settling is detected (timestamps are stable from the start).
    """
    diffs = np.diff(ptp_times)
    for i in range(len(diffs) - window + 1):
        if all(
            abs(diffs[i + j] - frame_period_ns) < frame_period_ns * tol
            for j in range(window)
        ):
            return i + 1
    return 0


def compute_alignment(
    all_ptp_times: list[np.ndarray],
    source_fps: float,
    target_fps: int | None,
) -> tuple[np.ndarray, list[int], int]:
    """Find the common recording window across cameras using PTP timestamps.

    Each camera may have started and stopped at a different absolute time.
    This function finds the PTP window during which all cameras were recording,
    then computes the video frame index each camera must seek to in order to
    begin at that shared moment.

    The returned ptp_times array is derived from camera 0 and saved identically
    into every camera's zarr group, so zarr frame t has the same PTP timestamp
    (i.e. the same absolute moment) across all cameras.

    Args:
        all_ptp_times:  ptp_frame_time arrays, one per camera, in video-frame order.
        source_fps:     acquisition frame rate (same for all cameras).
        target_fps:     desired output fps, or None to keep the source rate.

    Returns:
        aligned_ptp_ns:  PTP timestamps (ns) for each output frame, shape (T,).
                         Saved as the 'ptp_time_ns' coordinate in every zarr group.
        seek_frames:     for each camera, the 0-based video frame index whose PTP
                         time is closest to aligned_ptp_ns[0].
        output_fps:      fps to pass to the ffmpeg filter.
    """
    frame_period_ns = 1e9 / source_fps

    # Skip PTP settling frames at the start of each camera (the PTP clock takes
    # ~80-100 frames to lock; ptp[0] is unreliable and must not be used as the
    # sync reference).
    stable_idxs: list[int] = []
    for i, pt in enumerate(all_ptp_times):
        s = _find_stable_ptp_start(pt, frame_period_ns)
        stable_idxs.append(s)
        if s > 0:
            logger.warning(
                f"Camera {i}: first {s} frames have unstable PTP timestamps "
                f"(clock still locking). Using frame {s} "
                f"(PTP={pt[s] / 1e9:.3f}s) as the sync reference."
            )

    common_start_ptp = max(int(pt[s]) for pt, s in zip(all_ptp_times, stable_idxs))
    common_end_ptp = min(int(pt[-1]) for pt in all_ptp_times)

    overlap_sec = (common_end_ptp - common_start_ptp) / 1e9
    if overlap_sec <= 0:
        starts = [int(pt[0]) for pt in all_ptp_times]
        ends = [int(pt[-1]) for pt in all_ptp_times]
        raise ValueError(
            f"Cameras have no overlapping PTP window.\n"
            f"  PTP starts: {starts}\n"
            f"  PTP ends:   {ends}"
        )

    logger.info(
        f"Common PTP window: {overlap_sec:.1f}s "
        f"({round(overlap_sec * source_fps):.0f} frames at {source_fps:.4g} fps)"
    )

    # For each camera find the video frame closest to common_start_ptp.
    seek_frames: list[int] = []
    for i, pt in enumerate(all_ptp_times):
        idx = int(np.argmin(np.abs(pt - common_start_ptp)))
        error_ms = abs(int(pt[idx]) - common_start_ptp) / 1e6
        if error_ms > 0.5 * frame_period_ns / 1e6:  # more than half a frame off
            logger.warning(
                f"Camera {i}: nearest frame to sync start is {error_ms:.2f}ms off "
                f"(frame period = {frame_period_ns / 1e6:.2f}ms). "
                f"PTP jitter or dropped frame near sync boundary."
            )
        seek_frames.append(idx)
        logger.info(
            f"Camera {i}: seek to video frame {idx} (PTP alignment error: {error_ms:.3f}ms)"
        )

    # Use camera 0's PTP values as the output time coordinate.
    # Slice from its seek frame to the frame closest to common_end_ptp.
    ref_pt = all_ptp_times[0]
    ref_start = seek_frames[0]
    ref_end = int(np.argmin(np.abs(ref_pt - common_end_ptp)))
    aligned_ptp_ns = ref_pt[ref_start : ref_end + 1].copy()

    if target_fps is not None:
        step = round(source_fps / target_fps)
        if abs(source_fps / target_fps - step) > 0.01:
            logger.warning(
                f"source_fps ({source_fps:.4g}) / target_fps ({target_fps}) = "
                f"{source_fps / target_fps:.3f} is not a clean integer; "
                f"using step={step}."
            )
        aligned_ptp_ns = aligned_ptp_ns[::step]
        output_fps = target_fps
        logger.info(
            f"Downsampling {source_fps:.4g} → {target_fps} fps "
            f"(step={step}): {len(aligned_ptp_ns)} frames per camera"
        )
    else:
        output_fps = round(source_fps)
        logger.info(
            f"No downsampling: {len(aligned_ptp_ns)} frames per camera at {output_fps} fps"
        )

    return aligned_ptp_ns, seek_frames, output_fps


def _group_name_from_video(video_path: str) -> str:
    """Derive zarr group name from video stem. e.g. 'cam_1.mp4' -> 'cam_1'."""
    return Path(video_path).stem


def _count_frames(video_path: str) -> int:
    """Count video frames by scanning packets (reliable for raw HEVC streams)."""
    logger.info("Counting frames via packet scan (no container metadata found)...")
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-count_packets",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_read_packets",
        "-of",
        "csv=p=0",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return int(result.stdout.strip())


def get_video_info(video_path: str) -> dict:
    """Get video metadata using ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration",
        "-of",
        "json",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)["streams"][0]

    num, den = map(int, info["r_frame_rate"].split("/"))
    fps = num / den

    avg_num, avg_den = map(int, info["avg_frame_rate"].split("/"))
    avg_fps = avg_num / avg_den if avg_den != 0 else fps

    nb_frames_raw = info.get("nb_frames")
    duration_raw = info.get("duration")

    nb_frames = int(nb_frames_raw) if nb_frames_raw not in (None, "N/A") else None
    duration = float(duration_raw) if duration_raw not in (None, "N/A") else None

    if nb_frames is None:
        nb_frames = _count_frames(video_path)
    if duration is None:
        duration = nb_frames / fps

    return {
        "width": int(info["width"]),
        "height": int(info["height"]),
        "fps": fps,
        "avg_fps": avg_fps,
        "nb_frames": nb_frames,
        "duration": duration,
    }


def read_frame_at(
    video_path: str, time_sec: float, width: int, height: int
) -> np.ndarray:
    """Read a single frame at a given timestamp using ffmpeg."""
    cmd = [
        "ffmpeg",
        "-ss",
        str(time_sec),
        "-i",
        video_path,
        "-frames:v",
        "1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-v",
        "error",
        "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True, check=True)
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(height, width, 3)


def preview_crop(
    video_path: str,
    crop_x: int,
    crop_y: int,
    crop_w: int | None,
    crop_h: int | None,
    time_sec: float = 60.0,
) -> None:
    """Save annotated full frame and cropped frame as PNGs for inspection."""
    info = get_video_info(video_path)
    W, H = info["width"], info["height"]
    logger.info(f"Video: {W}x{H}, {info['fps']} fps, {info['nb_frames']} frames")

    if crop_w is None:
        crop_w = W - crop_x
    if crop_h is None:
        crop_h = H - crop_y

    assert crop_x >= 0 and crop_y >= 0, "Crop offsets must be non-negative"
    assert crop_x + crop_w <= W, f"Crop exceeds width: {crop_x}+{crop_w} > {W}"
    assert crop_y + crop_h <= H, f"Crop exceeds height: {crop_y}+{crop_h} > {H}"

    frame = read_frame_at(video_path, time_sec, W, H)
    annotated = frame.copy()
    red, bw = np.array([255, 0, 0], dtype=np.uint8), 4
    annotated[crop_y : crop_y + bw, crop_x : crop_x + crop_w] = red
    annotated[crop_y + crop_h - bw : crop_y + crop_h, crop_x : crop_x + crop_w] = red
    annotated[crop_y : crop_y + crop_h, crop_x : crop_x + bw] = red
    annotated[crop_y : crop_y + crop_h, crop_x + crop_w - bw : crop_x + crop_w] = red
    imsave("preview_full.png", annotated)
    logger.info("Saved preview_full.png (full frame with crop box)")

    cropped = frame[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]
    imsave("preview_cropped.png", cropped)
    logger.info(f"Saved preview_cropped.png ({crop_w}x{crop_h})")


def extract_frames(
    video_path: str,
    zarr_path: str,
    group_name: str,
    crop_x: int,
    crop_y: int,
    crop_w: int | None,
    crop_h: int | None,
    source_fps: float,
    output_fps: int,
    grayscale: bool,
    seek_frame: int = 0,
    aligned_ptp_ns: np.ndarray | None = None,
    timestamps_json: str = "",
) -> None:
    """Extract cropped frames from video and write to zarr.

    Args:
        source_fps:       acquisition frame rate of the video.
        output_fps:       fps passed to the ffmpeg filter (< source_fps if downsampling).
        seek_frame:       0-based video frame index to start extraction from.
                          Computed by compute_alignment() so all cameras begin at the
                          same absolute PTP time.
        aligned_ptp_ns:   PTP wall-clock timestamps (ns) for each output frame.
                          Derived from camera 0 and saved identically into every camera
                          group, so zarr frame t has the same PTP time across all cameras.
        timestamps_json:  source JSON path, stored in zarr attrs only.
    """
    info = get_video_info(video_path)
    W, H = info["width"], info["height"]

    if crop_w is None:
        crop_w = W - crop_x
    if crop_h is None:
        crop_h = H - crop_y

    nb_output_frames = (
        len(aligned_ptp_ns)
        if aligned_ptp_ns is not None
        else math.ceil(info["duration"] * output_fps)
    )
    t_seek = seek_frame / source_fps

    logger.info(
        f"Video: {W}x{H}, {source_fps:.4g} fps, {info['nb_frames']} frames, {info['duration']:.1f}s"
    )
    logger.info(f"Crop region: x={crop_x}, y={crop_y}, w={crop_w}, h={crop_h}")
    logger.info(
        f"Output: {output_fps} fps, {'grayscale' if grayscale else 'RGB'}, {nb_output_frames} frames"
    )
    logger.info(f"Seek to video frame {seek_frame} (t={t_seek:.4f}s)")

    assert crop_x + crop_w <= W, f"Crop exceeds width: {crop_x}+{crop_w} > {W}"
    assert crop_y + crop_h <= H, f"Crop exceeds height: {crop_y}+{crop_h} > {H}"

    num_channels = 1 if grayscale else 3

    vf_parts = []
    if not (crop_x == 0 and crop_y == 0 and crop_w == W and crop_h == H):
        vf_parts.append(f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y}")
    vf_parts.append(f"fps={output_fps}")
    vf_filters = ",".join(vf_parts)

    # Use slow seek (-ss after -i): decodes from the beginning up to t_seek.
    # Fast seek (-ss before -i) fails for raw HEVC streams which have no keyframes.
    cmd = ["ffmpeg", "-i", video_path]
    if t_seek > 0:
        cmd += ["-ss", f"{t_seek:.6f}"]
    cmd += [
        "-vf",
        vf_filters,
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-v",
        "error",
        "-stats",
        "pipe:1",
    ]

    axis_names = ("c", "t", "y", "x")
    zarr_shape = (num_channels, nb_output_frames, crop_h, crop_w)
    chunk_shape = (num_channels, 1, crop_h, crop_w)
    logger.info(f"Zarr shape: {zarr_shape}, chunks: {chunk_shape}, dtype: uint8")

    container = zarr.open(zarr_path, mode="a")
    grp = container.require_group(group_name)
    ds = grp.create_dataset(
        DATASET_NAME,
        shape=zarr_shape,
        chunks=chunk_shape,
        dtype=np.uint8,
        compressor=None,
        overwrite=True,
    )
    ds.attrs.update(
        {
            "resolution": (1,) * (len(axis_names) - 1),
            "offset": (0,) * (len(axis_names) - 1),
            "axis_names": axis_names,
            "source_fps": source_fps,
            "output_fps": output_fps,
            "source_video": video_path,
            "crop_region": {"x": crop_x, "y": crop_y, "w": crop_w, "h": crop_h},
            "seek_frame": seek_frame,
            "timestamps_json": timestamps_json,
        }
    )

    frame_bytes = crop_w * crop_h * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    frame_idx = 0
    pbar = tqdm(total=nb_output_frames, desc="Extracting frames", unit="frame")
    try:
        while True:
            raw = proc.stdout.read(frame_bytes)
            if len(raw) == 0:
                break
            if len(raw) < frame_bytes:
                logger.warning(
                    f"Incomplete frame {frame_idx} ({len(raw)}/{frame_bytes} bytes), stopping."
                )
                break
            if frame_idx >= nb_output_frames:
                break

            frame = np.frombuffer(raw, dtype=np.uint8).reshape(crop_h, crop_w, 3)
            if grayscale:
                frame_cyx = frame[:, :, 0][np.newaxis, ...]
            else:
                frame_cyx = np.moveaxis(frame, -1, 0)

            ds[:, frame_idx] = frame_cyx
            frame_idx += 1
            pbar.update(1)
    finally:
        pbar.close()
        proc.stdout.close()
        proc.wait()

    if frame_idx < nb_output_frames:
        logger.warning(
            f"Expected {nb_output_frames} frames but only got {frame_idx}; trimming."
        )
        ds.resize(num_channels, frame_idx, crop_h, crop_w)

    # Save PTP timestamps as the time coordinate.
    # aligned_ptp_ns is the same array for every camera, so ptp_time_ns[t] is
    # the same absolute wall-clock moment across all groups.
    if aligned_ptp_ns is not None:
        actual_ptp = aligned_ptp_ns[:frame_idx]
        grp.create_dataset(
            PTP_DATASET_NAME, data=actual_ptp, dtype=np.int64, overwrite=True
        )
        grp[PTP_DATASET_NAME].attrs.update(
            {
                "description": (
                    "PTP wall-clock timestamp (nanoseconds) for each output frame. "
                    "Derived from the reference camera and saved identically in every "
                    "camera group. Frame t in 'img' corresponds to the same absolute "
                    "moment across all cameras: ptp_time_ns[t] is the same value everywhere."
                ),
                "units": "nanoseconds",
                "source": timestamps_json,
            }
        )
        logger.info(
            f"Saved PTP time coordinate: {frame_idx} frames, "
            f"{(actual_ptp[-1] - actual_ptp[0]) / 1e9:.1f}s span"
        )

    logger.info(
        f"Done. {frame_idx} frames written to {zarr_path}/{group_name}/{DATASET_NAME}"
    )
    logger.info(f"Dataset shape: {ds.shape}, dtype: {ds.dtype}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract cropped frames from video to zarr."
    )
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--preview-time", type=float, default=60.0)
    parser.add_argument("--crop-x", type=int, default=0)
    parser.add_argument("--crop-y", type=int, default=0)
    parser.add_argument("--crop-w", type=int, default=None)
    parser.add_argument("--crop-h", type=int, default=None)
    parser.add_argument(
        "--target-fps",
        type=int,
        default=None,
        help=(
            "Target output fps after downsampling (must be <= source fps, "
            "ideally an integer divisor). Subsampling is applied consistently "
            "so zarr frame t remains the same moment in every camera group."
        ),
    )
    parser.add_argument("--grayscale", action="store_true", default=True)
    parser.add_argument("--rgb", action="store_true")
    parser.add_argument("--video", type=str, nargs="+", required=True)
    parser.add_argument("--zarr", type=str, default=None)
    parser.add_argument("--group", type=str, default=None)
    parser.add_argument(
        "--timestamps",
        type=str,
        nargs="+",
        required=True,
        help=(
            "Timestamp JSON file(s) in the same order as --video. "
            "Must be provided explicitly; auto-discovery is not supported. "
            "Cameras are aligned via ptp_frame_time (PTP wall-clock) so that "
            "zarr frame t is the same absolute moment across every group. "
            "Source fps is derived automatically from cam_frame_time."
        ),
    )

    args = parser.parse_args()

    if args.group is not None and len(args.video) > 1:
        parser.error("--group can only be used with a single --video.")

    if len(args.timestamps) != len(args.video):
        parser.error(
            f"--timestamps has {len(args.timestamps)} file(s) but --video has {len(args.video)}; "
            "they must match exactly."
        )

    if args.zarr is None:
        video_path = Path(args.video[0]).resolve()
        args.zarr = str(video_path.parent / (video_path.parent.name + ".zarr"))
        logger.info(f"Zarr path: '{args.zarr}'")

    grayscale = not args.rgb

    if args.preview:
        for video in args.video:
            preview_crop(
                video,
                args.crop_x,
                args.crop_y,
                args.crop_w,
                args.crop_h,
                args.preview_time,
            )
        return

    # -------------------------------------------------------------------------
    # PTP-based alignment across all cameras, then extract
    # -------------------------------------------------------------------------
    logger.info("Loading all timestamp JSONs...")
    all_ptp_times: list[np.ndarray] = []
    all_ts_fps: list[float] = []

    for ts_json in args.timestamps:
        ptp_times, ts_fps = load_timestamps(ts_json)
        if ptp_times is None:
            parser.error(
                f"Timestamp file '{ts_json}' has no ptp_frame_time field. "
                "PTP timestamps are required for multi-camera alignment."
            )
        if ts_fps is None:
            parser.error(
                f"Timestamp file '{ts_json}' has no cam_frame_time field. "
                "cam_frame_time is required to derive source fps."
            )
        all_ptp_times.append(ptp_times)
        all_ts_fps.append(ts_fps)

    source_fps = float(np.median(all_ts_fps))
    logger.info(
        f"Source fps from cam_frame_time (median across cameras): {source_fps:.4f} Hz"
    )

    if args.target_fps is not None and args.target_fps > source_fps:
        parser.error(
            f"--target-fps ({args.target_fps}) cannot exceed source fps ({source_fps:.4g})"
        )

    aligned_ptp_ns, seek_frames, output_fps = compute_alignment(
        all_ptp_times, source_fps, args.target_fps
    )

    for i, video in enumerate(args.video):
        group_name = args.group if args.group else _group_name_from_video(video)
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Processing: {video} -> {args.zarr}/{group_name}/{DATASET_NAME}")
        logger.info(f"Timestamps: {args.timestamps[i]}")
        logger.info(f"{'=' * 60}")
        extract_frames(
            video,
            args.zarr,
            group_name,
            args.crop_x,
            args.crop_y,
            args.crop_w,
            args.crop_h,
            source_fps,
            output_fps,
            grayscale,
            seek_frame=seek_frames[i],
            aligned_ptp_ns=aligned_ptp_ns,
            timestamps_json=args.timestamps[i],
        )


if __name__ == "__main__":
    main()
