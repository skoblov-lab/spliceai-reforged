import sys
from distutils.core import setup
from setuptools import find_packages

if sys.version_info < (3, 7, 0):
    print('This package requires Python >= 3.7.0')
    sys.exit(1)

setup(
    name='spliceai-reforged',
    version='0.1dev1',
    description='',
    author='Ilia Korvigo',
    author_email='',
    url='',
    packages=find_packages(),
    package_data={
        'splicelib.builtin_annotations': [
            'grch37.txt',
            'grch38.txt'
        ],
        'splicelib.trained_models': [
            'spliceai1.h5',
            'spliceai2.h5',
            'spliceai3.h5',
            'spliceai4.h5',
            'spliceai5.h5'
        ]
    },
    install_requires=[
        "importlib_resources ; python_version<'3.9'",
        'click>=7.1.2',
        'numpy>=1.18.1',
        'pandas>=1.0.3',
        'tensorflow>=2.0',
        'pysam>=0.15.3',
        'pyfaidx>=0.5.9',
        'ncls>=0.0.53'
    ],
    scripts=['spliceai.py']
)
