"""
Setup script for building Frigate's Cython extensions.
Run: python3 setup_cython.py build_ext --inplace
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# Cythonize all .pyx files with compiler directives
setup(
    name="frigate-cython",
    packages=["frigate.util", "frigate.embeddings", "frigate.events"],
    ext_modules=cythonize(
        [
            "frigate/util/image_cython.pyx",
            "frigate/util/object_cython.pyx",
            "frigate/util/multiprocessing_sync_cython.pyx",
            "frigate/embeddings/util_cython.pyx",
            "frigate/events/audio_cython.pyx",
            "frigate/detectors/detection_cython.pyx",
            "frigate/record/record_cython.pyx",
        ],
        language_level=3,
        compiler_directives={
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
        },
    ),
    include_dirs=[np.get_include()],
    zip_safe=False,
)
