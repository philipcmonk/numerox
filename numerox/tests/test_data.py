import numpy as np
from nose.tools import ok_

import pandas as pd

from numerox.data import Data
from numerox.testing import load_play_data, shares_memory


def micro_data(index=None, nfeatures=3):
    cols = ['era', 'region']
    cols += ['x' + str(i) for i in range(1, nfeatures + 1)]
    cols += ['y']
    df = pd.DataFrame(columns=cols)
    df.loc['index0'] = ['era1', 'train'] + [0.0] * nfeatures + [0.]
    df.loc['index1'] = ['era2', 'train'] + [0.1] * nfeatures + [1.]
    df.loc['index2'] = ['era2', 'train'] + [0.2] * nfeatures + [0.]
    df.loc['index3'] = ['era3', 'valuation'] + [0.3] * nfeatures + [1.]
    df.loc['index4'] = ['era3', 'valuation'] + [0.4] * nfeatures + [0.]
    df.loc['index5'] = ['era3', 'valuation'] + [0.5] * nfeatures + [1.]
    df.loc['index6'] = ['era4', 'valuation'] + [0.6] * nfeatures + [0.]
    df.loc['index7'] = ['eraX', 'test'] + [0.7] * nfeatures + [1.]
    df.loc['index8'] = ['eraX', 'test'] + [0.8] * nfeatures + [0.]
    df.loc['index9'] = ['eraX', 'live'] + [0.9] * nfeatures + [1.]
    if index is not None:
        df = df.iloc[index]
    data = Data(df)
    return data


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
