#!/usr/bin/env python

import os
from setuptools import setup, find_packages


SHORT_DESCRIPTION = ("Numerox is a Numerai competition toolbox written in "
                     "Python")


def get_long_description():
    with open('readme.rst', 'r') as fid:
        long_description = fid.read()
    idx = max(0, long_description.find("Numerox is a Numerai"))
    long_description = long_description[idx:]
    return long_description


def get_version_str():
    ver_file = os.path.join('numerox', 'version.py')
    with open(ver_file, 'r') as fid:
        version = fid.read()
    version = version.split("= ")
    version = version[1].strip()
    version = version.strip("\"")
    return version


CLASSIFIERS = [
    "Development Status :: 3 - Alpha",
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: BSD License",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Programming Language :: Python :: 2",
    "Programming Language :: Python :: 3",
    "Topic :: Scientific/Engineering"]


REQUIRES = ['numpy',
            'pandas',
            'tables',
            'scikit-learn',
            'requests',
            'nose']


metadata = dict(name='numerox',
                maintainer="Keith Goodman",
                description=SHORT_DESCRIPTION,
                long_description=get_long_description(),
                url="https://github.com/kwgoodman/numerox",
                license="Simplified BSD",
                classifiers=CLASSIFIERS,
                platforms="OS Independent",
                version=get_version_str(),
                packages=find_packages(),
                package_data={'numerox': ['LICENSE', 'readme.rst',
                              'release.rst', 'tests/test_data.hdf']},
                install_requires=REQUIRES)
setup(**metadata)
