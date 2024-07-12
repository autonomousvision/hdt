from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("cython_attention_mask.pyx", annotate=True),
)