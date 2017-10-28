import numpy as np


def row_sample(data, fraction=0.01, seed=0):
    "Randomly sample `fraction` of each era's rows; y is likely unbalanced"
    rs = np.random.RandomState(seed)
    era = data.era
    bool_idx = np.zeros(len(data), np.bool)
    eras = data.unique_era()
    for e in eras:
        idx = era == e
        n = idx.sum()
        nfrac = int(fraction * n)
        idx = np.where(idx)[0]
        rs.shuffle(idx)
        idx = idx[:nfrac]
        bool_idx[idx] = 1
    frac_data = data[bool_idx]
    return frac_data
