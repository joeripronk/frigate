"""
Setup script for building Frigate's detection_runners Cython extensions.
Run: python3 setup_cython_detectors.py build_ext --inplace
"""

from setuptools import setup
from Cython.Build import cythonize

# Cythonize all .pyx files in frigate/detectors with compiler directives
setup(
    name="frigate-detection-cython",
    packages=["frigate.detectors"],
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
