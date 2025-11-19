# setup.py
from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize

ext = Extension(
    name="convlib",
    sources=["convlib_cython.pyx", "convlib.c"],
    extra_compile_args=["-mavx2", "-mfma", "-O3"],
    extra_link_args=["-lopenblas"]
)

setup(
    name="convlib",
    ext_modules=cythonize([ext], language_level="3"),
)