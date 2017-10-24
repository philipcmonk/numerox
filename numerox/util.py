import numpy as np
from sklearn.model_selection import KFold


def cv(data, kfold=5, random_state=None):
    "Cross validation iterator that yields train, test data across eras"
    kf = KFold(n_splits=kfold, shuffle=True, random_state=random_state)
    eras = data.unique_era()
    for train_index, test_index in kf.split(eras):
        era_train = [eras[i] for i in train_index]
        era_test = [eras[i] for i in test_index]
        dtrain = data.era_isin(era_train)
        dtest = data.era_isin(era_test)
        yield dtrain, dtest


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
