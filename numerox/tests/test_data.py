import os

import numerox as nx

TEST_ARCHIVE = os.path.join(os.path.dirname(__file__), 'test_data.hdf')


def test_load():
    "make sure load runs without crashing"
    nx.load(TEST_ARCHIVE)
