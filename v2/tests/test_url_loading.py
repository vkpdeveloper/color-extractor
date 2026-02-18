from unittest.mock import MagicMock, patch
import io
import numpy as np
from PIL import Image
import pytest
import requests
from v2.src.color_pipeline.io import read_image_rgb

def test_read_image_rgb_url_success():
    # Create a dummy image
    img = Image.new('RGB', (10, 10), color='red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_bytes = img_byte_arr.getvalue()

    with patch('requests.get') as mock_get:
        # Mock the response
        mock_response = MagicMock()
        mock_response.content = img_bytes
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # Test
        url = "http://example.com/image.png"
        result = read_image_rgb(url)

        # Verify
        mock_get.assert_called_once_with(url, timeout=10)
        assert isinstance(result, np.ndarray)
        assert result.shape == (10, 10, 3)
        # Check if the pixel is red (255, 0, 0)
        assert np.all(result[0, 0] == [255, 0, 0])

def test_read_image_rgb_url_failure():
    with patch('requests.get') as mock_get:
        # Mock a 404 error
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Client Error")
        mock_get.return_value = mock_response

        # Test
        url = "http://example.com/nonexistent.png"
        with pytest.raises(requests.exceptions.HTTPError):
            read_image_rgb(url)

def test_read_image_rgb_local_file(tmp_path):
    # Create a dummy image file
    img = Image.new('RGB', (10, 10), color='blue')
    img_path = tmp_path / "test_image.png"
    img.save(img_path)

    # Test
    result = read_image_rgb(str(img_path))

    # Verify
    assert isinstance(result, np.ndarray)
    assert result.shape == (10, 10, 3)
    # Check if the pixel is blue (0, 0, 255)
    assert np.all(result[0, 0] == [0, 0, 255])
