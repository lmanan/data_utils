import argparse
import logging
from pathlib import Path
from zipfile import ZipFile

import requests
import gdown

logging.basicConfig(level=logging.INFO)


def is_google_drive_resource(zip_url: str) -> bool:
    """
    Heuristic to decide whether the given string is a Google Drive URL or file ID.
    """
    if zip_url.startswith("http"):
        return "drive.google.com" in zip_url or "docs.google.com" in zip_url
    # If it does not look like a URL at all, assume it is a Drive file ID
    return True


def download_with_requests(url: str, output_zip: Path) -> None:
    """
    Download a URL using requests and save to output_zip.
    Handles Dropbox 'dl=0' → 'dl=1' conversion.
    """
    # Force direct Dropbox download if needed
    if "dropbox.com" in url:
        url = url.replace("dl=0", "dl=1")

    logging.info(f"Downloading from URL with requests: {url}")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with output_zip.open("wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)


def download_with_gdown(zip_url: str, output_zip: Path) -> None:
    """
    Download from Google Drive using gdown (URL or file ID).
    """
    logging.info(f"Downloading from Google Drive with gdown: {zip_url}")
    # gdown supports both URL and file ID
    gdown.download(zip_url, str(output_zip), quiet=False)


def extract_data(zip_url: str, data_dir: str) -> None:
    """
    Downloads and extracts a zip file from:
      - Google Drive (via gdown)
      - Any HTTP(S) URL (via requests: Dropbox, S3, GitHub, etc.)

    Parameters
    ----------
    zip_url : str
        Google Drive file ID / URL OR any HTTP(S) URL of the zip file.
    data_dir : str
        Path to the directory where data will be stored.

    Returns
    -------
    None
    """
    target_path = Path(data_dir)

    if target_path.exists():
        logging.info(f"Directory already exists at {target_path}")
        return

    target_path.mkdir(parents=True, exist_ok=True)
    logging.info(f"Created directory {target_path}")

    output_zip = target_path / "data.zip"

    try:
        # Decide which downloader to use
        if is_google_drive_resource(zip_url):
            download_with_gdown(zip_url, output_zip)
        else:
            download_with_requests(zip_url, output_zip)

        # Extract zip
        logging.info(f"Extracting {output_zip} ...")
        with ZipFile(output_zip, "r") as zfile:
            zfile.extractall(target_path)

        logging.info(f"Downloaded and extracted data to {target_path}")

    except Exception as e:
        logging.error(f"Failed to download or extract data: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Download and extract a zip file from Google Drive or any URL."
    )
    parser.add_argument(
        "--zip_url",
        type=str,
        required=True,
        help="Google Drive file ID/URL or any HTTP(S) URL of the zip file",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Directory where data will be stored (default: ./data)",
    )

    args = parser.parse_args()
    extract_data(args.zip_url, args.data_dir)


if __name__ == "__main__":
    main()
