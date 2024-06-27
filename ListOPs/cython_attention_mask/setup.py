from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("cython_attention_mask_blue.pyx", annotate=True),
)