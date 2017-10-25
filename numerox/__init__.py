# flake8: noqa

from numerox.data import Data, load, load_zip, concat
from numerox.util import cv, row_sample, shares_memory
from numerox.version import __version__

try:
    from numpy.testing import Tester
    test = Tester().test
    del Tester
except (ImportError, ValueError):
    print("No numerox unit testing available")
