import numpy as np
from nose.tools import ok_

from numerox import Prediction
from numerox.testing import load_play_data, shares_memory


def test_prediction_copies():
    "prediction properties should be copies"

    d = load_play_data()
    p = Prediction()
    p.append(d.ids, d.y)

    ok_(shares_memory(p, p), "looks like shares_memory failed")
    ok_(~shares_memory(p, p.copy()), "should be a copy")

    ok_(~shares_memory(p, p.ids), "p.ids should be a copy")
    ok_(~shares_memory(p, p.yhat), "p.yhat should be a copy")


def test_data_properties():
    "prediction properties should not be corrupted"

    d = load_play_data()
    p = Prediction()
    p.append(d.ids, d.y)

    ok_((p.ids == p.df.index).all(), "ids is corrupted")
    ok_((p.ids == d.df.index).all(), "ids is corrupted")
    idx = ~np.isnan(p.df.yhat)
    ok_((p.yhat[idx] == p.df.yhat[idx]).all(), "yhat is corrupted")
    ok_((p.yhat[idx] == d.df.y[idx]).all(), "yhat is corrupted")
