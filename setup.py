from setuptools import setup, find_packages

setup(name="mccepy", 
      version=0.1.0,
      author="Annabelle Redelmeier",
      author_email="anr@nr.no",
      description="Python implementation of MCCE: Monte Carlo Generateing of Counterfactual Explanations",
      url="https://github.com/NorskRegnesentral/mccepy",
      packages=find_packages(),
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
      ],
      python_requires='>=3.6'
      )