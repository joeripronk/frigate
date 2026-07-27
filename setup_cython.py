"""
Setup script for building Frigate's Cython extensions.
Run: python3 setup_cython.py build_ext --inplace
"""

from setuptools import setup
from Cython.Build import cythonize

# Cythonize all .pyx files with compiler directives
setup(
    name="frigate-cython",
    packages=["frigate.util", "frigate.embeddings"],
    ext_modules=cythonize(
        [
            "frigate/util/image_cython.pyx",
            "frigate/embeddings/util_cython.pyx",
        ],
        language_level=3,
        compiler_directives={
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
        },
    ),
    zip_safe=False,
)
