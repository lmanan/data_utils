from data_utils.extract_data import is_google_drive_resource


class TestIsGoogleDriveResource:
    """Tests for the is_google_drive_resource() function."""

    def test_google_drive_url(self):
        """Google Drive URLs should be detected."""
        assert (
            is_google_drive_resource("https://drive.google.com/file/d/abc123") is True
        )
        assert is_google_drive_resource("https://drive.google.com/uc?id=xyz789") is True

    def test_dropbox_url(self):
        """Dropbox URLs should not be detected as Drive resources."""
        assert (
            is_google_drive_resource("https://www.dropbox.com/s/abc/file.zip?dl=0")
            is False
        )

    def test_github_url(self):
        """GitHub URLs should not be detected as Drive resources."""
        assert (
            is_google_drive_resource("https://github.com/user/repo/archive/main.zip")
            is False
        )

    def test_generic_http_url(self):
        """Generic HTTP URLs should not be detected as Drive resources."""
        assert is_google_drive_resource("https://example.com/data.zip") is False
        assert is_google_drive_resource("http://example.com/data.zip") is False
