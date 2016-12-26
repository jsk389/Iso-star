from distutils.core import setup
from Cython.Build import cythonize
import numpy as np
#setup(
#    ext_modules= cythonize('inference_fns_hierarch.pyx'),
#                 include_dirs=[np.get_include()]
#)

setup(
    ext_modules= cythonize('sini_inference_fns_mod.pyx'),
                 include_dirs=[np.get_include()]
)
