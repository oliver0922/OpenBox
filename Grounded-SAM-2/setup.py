"""Expose sam2 / sam2_configs as top-level packages (editable install).

Intentionally does NOT build the SAM2 CUDA extension (sam2._C): the stored
batches were generated without it, and enabling it changes the mask
postprocessing results. See README.md.
"""
from setuptools import setup, find_packages

setup(
    name='grounded-sam-2-waymo',
    version='1.0',
    packages=find_packages(include=['sam2', 'sam2.*', 'sam2_configs']),
    package_data={'sam2_configs': ['*.yaml']},
)
