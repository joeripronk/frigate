"""
Benchmarks for Cython-accelerated image processing functions.

Run: python3 -u -m unittest frigate.test.benchmark_image_cython
"""

import unittest
import numpy as np
import sys

sys.path.insert(0, '/home/joeri/src/frigate')

from frigate.util.image_cython import yuv_to_3_channel_yuv


class TestYUVFunctions(unittest.TestCase):
    """Basic tests for YUV functions."""

    def test_yuv_to_3_channel(self):
        """Test yuv_to_3_channel_yuv with proper 4:2:0 YUV data."""
        height = 720
        width = 1280
        
        # Create proper 4:2:0 interleaved YUV data
        yuv_frame = np.zeros((height * 3 // 2, width), dtype=np.uint8)
        yuv_frame[0:height, :] = 16  # Y plane
        uv_height = height // 2
        uv_width = width // 2
        yuv_frame[height:height+uv_height, 0:uv_width] = 128  # U plane
        yuv_frame[height:height+uv_height, uv_width:width] = 128  # V plane
        yuv_frame[height+uv_height:height+2*uv_height, 0:uv_width] = 128
        yuv_frame[height+uv_height:height+2*uv_height, uv_width:width] = 128

        result = yuv_to_3_channel_yuv(yuv_frame)
        self.assertEqual(result.shape, (height, width, 3))
        self.assertEqual(result[:, :, 0].sum(), 16 * height * width)
        self.assertEqual(result[:, :, 1].sum(), 128 * height * width)
        self.assertEqual(result[:, :, 2].sum(), 128 * height * width)


if __name__ == '__main__':
    unittest.main()
