from setuptools import setup, Extension
from Cython.Build import cythonize
from glob import glob
import numpy as np


cython_ext_modules = [
    Extension(
        "glmext._C._loss",
        sources=sorted(glob("glmext/_C/_loss.pyx")),
        language="c++",
        include_dirs=[np.get_include()],
        extra_compile_args=["-std=c++11"],
        extra_link_args=["-std=c++11"],
    )
]

ext_modules = cythonize(cython_ext_modules, build_dir="build")

setup(
    ext_modules=ext_modules
)
