import os

import numpy as np
from nose.tools import ok_

import numerox as nx
from numerox.testing import shares_memory

TEST_ARCHIVE = os.path.join(os.path.dirname(__file__), 'test_data.hdf')


def test_prediction_copies():
    "prediction properties should be copies"

    d = nx.load_data(TEST_ARCHIVE)
    p = nx.Prediction()
    p.append(d.ids, d.y)

    ok_(shares_memory(p, p), "looks like shares_memory failed")
    ok_(~shares_memory(p, p.copy()), "should be a copy")

    ok_(~shares_memory(p, p.ids), "p.ids should be a copy")
    ok_(~shares_memory(p, p.y), "p.y should be a copy")


def test_data_properties():
    "prediction properties should not be corrupted"

    d = nx.load_data(TEST_ARCHIVE)
    p = nx.Prediction()
    p.append(d.ids, d.y)

    ok_((p.ids == p.df.index).all(), "ids is corrupted")
    idx = ~np.isnan(p.df.y)
    ok_((p.y[idx] == p.df.y[idx]).all(), "y is corrupted")
