"""
Cython-accelerated tensor operations for detection runners.

This module provides Cython-optimized implementations of common tensor
operations used by detection runners (CUDA graph, OpenVINO, RKNN).
"""

# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

import numpy as np
from typing import Any

# Type declarations for tensor operations

def ascontiguousarray(arr: Any) -> np.ndarray:
    """Ensure array is contiguous in memory.
    
    Args:
        arr: Input array
        
    Returns:
        Contiguous copy or reference
    """
    return np.ascontiguousarray(arr)

def transpose_nchw_to_nhwc(data: np.ndarray) -> np.ndarray:
    """Transpose from NCHW to NHWC format.
    
    RKNN expects NHWC format, but ONNX typically provides NCHW.
    Transpose from [batch, channels, height, width] to [batch, height, width, channels].
    
    Args:
        data: Input tensor with shape (batch, channels, height, width)
        
    Returns:
        Transposed tensor with shape (batch, height, width, channels)
    """
    if data.ndim != 4:
        return data
    
    # Check if transpose is needed (channels != 1)
    if data.shape[1] == 1:
        return data
    
    # Perform transpose: (0, 2, 3, 1)
    return np.transpose(data, (0, 2, 3, 1))

def convert_dtype_inplace(data: np.ndarray, expected_dtype: np.dtype) -> np.ndarray:
    """Convert data in-place to expected dtype.
    
    Args:
        data: Input array
        expected_dtype: Target dtype
        
    Returns:
        Converted array
    """
    if data.dtype == expected_dtype:
        return data
    
    return data.astype(expected_dtype, copy=False)

def prepare_tensor(tensor_input: dict[str, Any], name: str) -> np.ndarray:
    """Prepare tensor for CUDA graph inference.
    
    Extracts tensor from input dict, ensures contiguity, and binds to CUDA.
    
    Args:
        tensor_input: Dictionary mapping input names to tensor data
        name: Input name key
        
    Returns:
        Contiguous tensor data
    """
    input_name = list(tensor_input.keys())[0]
    tensor_input = tensor_input[input_name]
    tensor_nd = ascontiguousarray(tensor_input)
    return tensor_nd

def nchw_to_nhwc_transpose(pixel_data: np.ndarray) -> np.ndarray:
    """Transpose pixel data from NCHW to NHWC format.
    
    Used by RKNNModelRunner for pixel_values input.
    
    Args:
        pixel_data: Input tensor with shape (batch, 3, height, width)
        
    Returns:
        Transposed tensor with shape (batch, height, width, 3)
    """
    if len(pixel_data.shape) == 4 and pixel_data.shape[1] == 3:
        return transpose_nchw_to_nhwc(pixel_data)
    return pixel_data

def nchw_to_nhwc_transpose_face(face_data: np.ndarray) -> np.ndarray:
    """Transpose face data from NCHW to NHWC format.
    
    Used by RKNNModelRunner for ArcFace data input.
    
    Args:
        face_data: Input tensor with shape (batch, 3, height, width)
        
    Returns:
        Transposed tensor with shape (batch, height, width, 3)
    """
    if len(face_data.shape) == 4 and face_data.shape[1] == 3:
        return transpose_nchw_to_nhwc(face_data)
    return face_data

def face_normalization(face_data: np.ndarray) -> np.ndarray:
    """Normalize face data to uint8 [0, 255].
    
    RKNN runtime applies mean=127.5/std=127.5 internally before first layer.
    Undo Python normalization to uint8 [0, 255].
    
    Args:
        face_data: Normalized face tensor (typically float32 in [0, 1])
        
    Returns:
        uint8 face data in range [0, 255]
    """
    normalized_nd = ((face_data + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return normalized_nd

def copyto_inplace(dst: np.ndarray, src: np.ndarray) -> None:
    """Copy source array into pre-allocated destination array in-place.
    
    Used to copy numpy arrays into OpenVINO/RKNN tensor buffers without
    allocating new memory.
    
    Args:
        dst: Destination numpy buffer
        src: Source array to copy from
    """
    np.copyto(dst, src, casting="unsafe")


# Cython extension setup - setuptools will call this during build
def setup():
    from setuptools import setup as setuptools_setup
    from Cython.Build import cythonize
    setuptools_setup(
        name="frigate-detection-runners-cython",
        ext_modules=cythonize(
            "frigate/detectors/detection_runners_cython.pyx",
            language_level=3,
            compiler_directives={
                "boundscheck": False,
                "wraparound": False,
                "cdivision": True,
            },
        ),
        zip_safe=False,
    )
