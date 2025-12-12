import numpy as np
import pytest
from PIL import Image

from data_utils.split_data import split_data, filter_matching_files, get_image_files


@pytest.fixture
def sample_data(tmp_path):
    """Creates a temporary directory with sample images and masks."""
    project_path = tmp_path / "test_project" / "download" / "all_data"
    image_dir = project_path / "images"
    mask_dir = project_path / "masks"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)

    # Create 20 sample image/mask pairs
    for i in range(20):
        # Create random image (64x64 RGB)
        img_arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        img = Image.fromarray(img_arr)
        img.save(image_dir / f"frame_{i:03d}.png")

        # Create corresponding mask (64x64 grayscale)
        mask_arr = np.random.randint(0, 5, (64, 64), dtype=np.uint8)
        mask = Image.fromarray(mask_arr)
        mask.save(mask_dir / f"frame_{i:03d}.png")

    return tmp_path, "test_project", "all_data"


class TestSplitData:
    def test_basic_split(self, sample_data):
        """Test that split produces correct number of files in each set."""
        data_dir, project_name, source_name = sample_data

        counts = split_data(
            data_dir=str(data_dir),
            project_name=project_name,
            source_name=source_name,
            test_fraction=0.2,  # 4 test
            val_fraction=0.25,  # 4 val (25% of remaining 16)
            seed=42,
        )

        assert counts["test"] == 4
        assert counts["val"] == 4
        assert counts["train"] == 12
        assert counts["train"] + counts["val"] + counts["test"] == 20

    def test_output_directories_created(self, sample_data):
        """Test that train/val/test directories are created with images and masks."""
        data_dir, project_name, source_name = sample_data

        split_data(
            data_dir=str(data_dir),
            project_name=project_name,
            source_name=source_name,
            test_fraction=0.2,
            val_fraction=0.2,
        )

        base_path = data_dir / project_name
        for split in ["train", "val", "test"]:
            assert (base_path / split / "images").exists()
            assert (base_path / split / "masks").exists()

    def test_files_saved_as_numpy(self, sample_data):
        """Test that output files are .npy format and loadable."""
        data_dir, project_name, source_name = sample_data

        split_data(
            data_dir=str(data_dir),
            project_name=project_name,
            source_name=source_name,
            test_fraction=0.2,
            val_fraction=0.2,
        )

        base_path = data_dir / project_name
        train_images = list((base_path / "train" / "images").glob("*.npy"))

        assert len(train_images) > 0
        assert all(f.suffix == ".npy" for f in train_images)

        # Verify array is loadable and has correct shape
        arr = np.load(train_images[0])
        assert arr.shape == (64, 64, 3)

    def test_image_mask_pairing_preserved(self, sample_data):
        """Test that image and mask filenames match in each split."""
        data_dir, project_name, source_name = sample_data

        split_data(
            data_dir=str(data_dir),
            project_name=project_name,
            source_name=source_name,
            test_fraction=0.2,
            val_fraction=0.2,
        )

        base_path = data_dir / project_name
        for split in ["train", "val", "test"]:
            image_stems = {f.stem for f in (base_path / split / "images").glob("*.npy")}
            mask_stems = {f.stem for f in (base_path / split / "masks").glob("*.npy")}
            assert image_stems == mask_stems

    def test_consecutive_mode(self, sample_data):
        """Test that consecutive mode produces contiguous blocks."""
        data_dir, project_name, source_name = sample_data

        split_data(
            data_dir=str(data_dir),
            project_name=project_name,
            source_name=source_name,
            test_fraction=0.2,
            val_fraction=0.25,
            consecutive=True,
        )

        base_path = data_dir / project_name

        # Get sorted frame numbers from each split
        def get_frame_numbers(split):
            files = (base_path / split / "images").glob("*.npy")
            return sorted(int(f.stem.split("_")[1]) for f in files)

        test_frames = get_frame_numbers("test")
        val_frames = get_frame_numbers("val")
        train_frames = get_frame_numbers("train")

        # Verify contiguous blocks: test should be first, then val, then train
        assert test_frames == list(range(0, 4))
        assert val_frames == list(range(4, 8))
        assert train_frames == list(range(8, 20))

    def test_reproducible_with_seed(self, sample_data):
        """Test that same seed produces same split."""
        data_dir, project_name, source_name = sample_data

        # First split
        split_data(
            data_dir=str(data_dir),
            project_name=project_name,
            source_name=source_name,
            test_fraction=0.2,
            val_fraction=0.2,
            seed=123,
        )

        base_path = data_dir / project_name
        first_train = {f.stem for f in (base_path / "train" / "images").glob("*.npy")}

        # Clean up and split again with same seed
        import shutil

        for split in ["train", "val", "test"]:
            shutil.rmtree(base_path / split)

        split_data(
            data_dir=str(data_dir),
            project_name=project_name,
            source_name=source_name,
            test_fraction=0.2,
            val_fraction=0.2,
            seed=123,
        )

        second_train = {f.stem for f in (base_path / "train" / "images").glob("*.npy")}

        assert first_train == second_train


class TestFilterMatchingFiles:
    def test_filters_unmatched_files(self, tmp_path):
        """Test that only matching image/mask pairs are kept."""
        image_dir = tmp_path / "images"
        mask_dir = tmp_path / "masks"
        image_dir.mkdir()
        mask_dir.mkdir()

        # Create images: 0, 1, 2, 3
        for i in range(4):
            (image_dir / f"img_{i}.png").touch()

        # Create masks: 1, 2, 3, 4 (0 missing, 4 extra)
        for i in range(1, 5):
            (mask_dir / f"img_{i}.png").touch()

        images, masks = filter_matching_files(
            get_image_files(image_dir),
            get_image_files(mask_dir),
        )

        assert len(images) == 3
        assert len(masks) == 3
        assert {p.stem for p in images} == {"img_1", "img_2", "img_3"}

    def test_matches_across_extensions(self, tmp_path):
        """Test that files match by stem even with different extensions."""
        image_dir = tmp_path / "images"
        mask_dir = tmp_path / "masks"
        image_dir.mkdir()
        mask_dir.mkdir()

        # Image as PNG, mask as TIF
        (image_dir / "frame_001.png").touch()
        (mask_dir / "frame_001.tif").touch()

        images, masks = filter_matching_files(
            get_image_files(image_dir),
            get_image_files(mask_dir),
        )

        assert len(images) == 1
        assert len(masks) == 1
        assert images[0].stem == masks[0].stem


class TestGetImageFiles:
    def test_finds_all_supported_formats(self, tmp_path):
        """Test that all supported image formats are found."""
        for ext in ["png", "jpg", "jpeg", "tif", "tiff"]:
            (tmp_path / f"image.{ext}").touch()

        files = get_image_files(tmp_path)
        assert len(files) == 5

    def test_finds_uppercase_extensions(self, tmp_path):
        """Test that uppercase extensions are also found."""
        (tmp_path / "image.PNG").touch()
        (tmp_path / "image.JPG").touch()

        files = get_image_files(tmp_path)
        assert len(files) == 2
