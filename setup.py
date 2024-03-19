from setuptools import setup, find_packages
from setuptools.extension import Extension
import numpy

setup(
    name='Diffusionanalysis',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'ase',
   #     'cython',
    ],
    author='Chris Davies',
    author_email='your.email@example.com',
    description='A library for performing diffusion analysis on atomic trajectories.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/chrisdavi3s/DiffusionAnalysis',
    classifiers=[
        # Add classifiers for your package
    ],
    #ext_modules=cythonize([Extension("*", ["*.pyx"])], include_dirs=[numpy.get_include()])
)