import os

import numpy as np

import numerox as nx

TEST_DATA = os.path.join(os.path.dirname(__file__), 'tests', 'test_data.hdf')


def shares_memory(data1, data_or_array2):
    "True if `data1` shares memory with `data_or_array2`; False otherwise"
    isdata_like = isinstance(data_or_array2, nx.Data)
    isdata_like = isdata_like or isinstance(data_or_array2, nx.Prediction)
    for col in data1._column_list():
        a1 = data1.df[col].values
        if isdata_like:
            a2 = data_or_array2.df[col].values
        else:
            a2 = data_or_array2
        if np.shares_memory(a1, a2):
            return True
    return False


def load_play_data():
    return nx.load_data(TEST_DATA)
