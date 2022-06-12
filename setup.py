# -*- coding: utf-8 -*-
"""
Setup file for foxes
"""
from setuptools import setup, find_packages

# grab version number from version file:
exec(open('foxes/version.py').read())

setup(name='foxes',
      version=__version__,
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
              'sphinx', 
              'sphinx_rtd_theme',
              'nbsphinx', 
          ],
      },
      zip_safe=True)