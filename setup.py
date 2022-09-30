from setuptools import setup, find_packages
from distutils.core import setup

setup(
  name="mcce", 
  version="0.1.0",
  description="Python implementation of MCCE: Monte Carlo Generateing of Counterfactual Explanations",
  packages=find_packages(),
  classifiers=[
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    ],
    python_requires='>=3.6'
    )