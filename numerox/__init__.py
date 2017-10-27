# flake8: noqa

# classes
from numerox.data import Data
from numerox.prediction import Prediction
from numerox.report import Report

# load
from numerox.data import load_data, load_zip
from numerox.testing import load_play_data
from numerox.report import load_report

# misc
from numerox.data import concat
from numerox.util import cv, row_sample
from numerox.version import __version__

try:
    from numpy.testing import Tester
    test = Tester().test
    del Tester
except (ImportError, ValueError):
    print("No numerox unit testing available")
