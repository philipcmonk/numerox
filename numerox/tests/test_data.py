import numpy as np
from nose.tools import ok_

from numerox.testing import load_play_data, shares_memory


def test_data_copies():
    "data properties should be copies"

    d = load_play_data()

    ok_(shares_memory(d, d), "looks like shares_memory failed")
    ok_(~shares_memory(d, d.copy()), "should be a copy")

    ok_(~shares_memory(d, d.ids), "d.ids should be a copy")
    ok_(~shares_memory(d, d.era), "d.era should be a copy")
    ok_(~shares_memory(d, d.region), "d.region should be a copy")
    ok_(~shares_memory(d, d.x), "d.x should be a copy")
    ok_(~shares_memory(d, d.y), "d.y should be a copy")


def test_data_properties():
    "data properties should not be corrupted"

    d = load_play_data()

    ok_((d.ids == d.df.index).all(), "ids is corrupted")
    ok_((d.era == d.df.era).all(), "era is corrupted")
    ok_((d.region == d.df.region).all(), "region is corrupted")

    idx = ~np.isnan(d.df.y)
    ok_((d.y[idx] == d.df.y[idx]).all(), "y is corrupted")

    x = d.x
    for i, name in enumerate(d._x_names()):
        ok_((x[:, i] == d.df[name]).all(), "%s is corrupted" % name)
