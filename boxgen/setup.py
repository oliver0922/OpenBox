"""Build the two vendored OpenPCDet CUDA ops the box generation uses.

Usage (from this directory, inside the `openbox` env)::

    TORCH_CUDA_ARCH_LIST="8.6" python setup.py build_ext --inplace   # 8.6 = your GPU's compute capability; omit the variable to autodetect

That drops ``iou3d_nms_cuda`` / ``roiaware_pool3d_cuda`` next to their sources
under ``pcdet/ops/``; afterwards ``PYTHONPATH=<this directory>`` serves both
``openbox_boxgen`` and ``pcdet``.  Sources are an unmodified subset of
OpenPCDet (Apache-2.0, https://github.com/open-mmlab/OpenPCDet).
"""
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='openbox-pcdet-ops',
    version='1.0',
    description='Vendored OpenPCDet CUDA ops (iou3d NMS, points-in-boxes) for OpenBox box generation',
    cmdclass={'build_ext': BuildExtension},
    ext_modules=[
        CUDAExtension(
            name='pcdet.ops.iou3d_nms.iou3d_nms_cuda',
            sources=['pcdet/ops/iou3d_nms/src/iou3d_cpu.cpp',
                     'pcdet/ops/iou3d_nms/src/iou3d_nms_api.cpp',
                     'pcdet/ops/iou3d_nms/src/iou3d_nms.cpp',
                     'pcdet/ops/iou3d_nms/src/iou3d_nms_kernel.cu'],
        ),
        CUDAExtension(
            name='pcdet.ops.roiaware_pool3d.roiaware_pool3d_cuda',
            sources=['pcdet/ops/roiaware_pool3d/src/roiaware_pool3d.cpp',
                     'pcdet/ops/roiaware_pool3d/src/roiaware_pool3d_kernel.cu'],
        ),
    ],
)
