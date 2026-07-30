"""
Setup script for building Frigate's Cython extensions.
Run: python3 setup_cython.py build_ext --inplace
"""

import numpy as np
from Cython.Build import cythonize
from setuptools import setup

# Cythonize all .pyx files with compiler directives
setup(
    name="frigate-cython",
    packages=[
        "frigate.util",
        "frigate.embeddings",
        "frigate.events",
        "frigate.motion",
        "frigate.video",
        "frigate.track",
        "frigate.record",
        "frigate.data_processing.common.face",
        "frigate.data_processing.common.license_plate",
    ],
    ext_modules=cythonize(
        [
            "frigate/util/image_cython.pyx",
            "frigate/util/model_cython.pyx",
            "frigate/util/object_cython.pyx",
            "frigate/util/multiprocessing_sync_cython.pyx",
            "frigate/embeddings/util_cython.pyx",
            "frigate/events/audio_cython.pyx",
            "frigate/detectors/detection_cython.pyx",
            "frigate/record/record_cython.pyx",
            "frigate/motion/motion_cython.pyx",
            "frigate/video/detect_cython.pyx",
            "frigate/track/tracking_cython.pyx",
            "frigate/data_processing/common/face/model_cython.pyx",
            "frigate/data_processing/common/license_plate/license_plate_cython.pyx",
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
