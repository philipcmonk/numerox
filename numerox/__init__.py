# flake8: noqa

from numerox.data import Data, load_data, load_zip, concat
from numerox.util import cv, row_sample, shares_memory
from numerox.prediction import Prediction, load_prediction
from numerox.version import __version__

try:
    from numpy.testing import Tester
    test = Tester().test
    del Tester
except (ImportError, ValueError):
    print("No numerox unit testing available")
