# -*- coding: utf-8 -*-
"""
Setup file for foxes
"""
import os
from setuptools import setup, find_packages

repo = os.path.dirname(__file__)
try:
    from git_utils import write_vers
    version = write_vers(vers_file='foxes/__init__.py', repo=repo, skip_chars=1)
except Exception:
    version = '999'

setup(name='foxes',
      version=version,
      description='Farm Optimization and eXtended yield Evaluation Software',
      long_description='README.md',
      url='https://github.com/FraunhoferIWES/foxes',
      project_urls={
          'Documentation': 'https://github.com/FraunhoferIWES/foxes/',
          'Changelog': 'https://github.com/FraunhoferIWES/foxes/notebooks/ChangeLog.html',
          'Source': 'https://github.com/FraunhoferIWES/foxes',
          'Tracker': 'https://github.com/FraunhoferIWES/foxes/-/issues',
      },
      author='Fraunhofer IWES',
      author_email='jonas.schmidt@iwes.fraunhofer.de',
      license='MIT',
      packages=find_packages(),
      package_data={
          'foxes': ['examples/*/*.csv',
                    'examples/field_data_nc/data/*.nc',
                    'tests/*/*/*/*.csv',
                    'tests/*/*/*/*.csv.gz',
                    'tests/verification/*/*/flappy/*.csv.gz',
                    ],
      },
      install_requires=[
        'matplotlib',
        'numpy',
        'xarray',
        'dask',
        'dask[distributed]',
        'scipy',
        'netcdf4',
      ],
      extras_require={
          'dev': [
              'gitutils',
              'unittest',
              'sphinx', 
              'sphinx_rtd_theme',
              'nbsphinx', 
          ],
      },
      zip_safe=True)