import zipfile

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


class Data(object):

    def __init__(self, ids, era, region, x, y):
        self.ids = ids
        self.era = era
        self.region = region
        self.x = x
        self.y = y

    def cv(self, kfold=5, random_state=None):
        kf = KFold(n_splits=kfold, shuffle=True, random_state=random_state)
        eras = np.unique(self.era)
        for train_index, test_index in kf.split(eras):
            idx = np.zeros(self.size, dtype=np.bool)
            for i in train_index:
                idx = np.logical_or(idx, self.era == eras[i])
            dtrain = self[idx]
            idx = np.zeros(self.size, dtype=np.bool)
            for i in test_index:
                idx = np.logical_or(idx, self.era == eras[i])
            dtest = self[idx]
            yield dtrain, dtest

    def __getitem__(self, index):

        typidx = type(index)

        if typidx is str:
            if index.startswith('era'):
                idx = self.era == index
            else:
                if index == 'train':
                    idx = self.region == 'train'
                elif index == 'validation':
                    idx = self.region == 'validation'
                elif index == 'test':
                    idx = self.region == 'test'
                elif index == 'live':
                    idx = self.region == 'live'
                elif index == 'tournament':
                    idx = self.region == 'validation'
                    idx = np.logical_or(idx, self.region == 'test')
                    idx = np.logical_or(idx, self.region == 'live')
        elif typidx is np.ndarray:
            idx = index
        else:
            raise IndexError('indexing type not recognized')

        d = Data(self.ids[idx],
                 self.era[idx],
                 self.region[idx],
                 self.x[idx],
                 self.y[idx])

        return d

    @property
    def size(self):
        return self.x.shape[0]


# ---------------------------------------------------------------------------
# load  helper functions

TRAIN_FILE = 'numerai_training_data.csv'
TOURN_FILE = 'numerai_tournament_data.csv'


def load_zip(dataset_zip):

    # load
    zf = zipfile.ZipFile(dataset_zip)
    header, ids, era, region, x, y = load_csv(zf.open(TRAIN_FILE))
    header2, ids2, era2, region2, x2, y2 = load_csv(zf.open(TOURN_FILE))

    # check headers
    header0 = expected_headers()
    if (header != header0).any():
        raise ValueError('train file column header not recognized')
    if (header2 != header0).any():
        raise ValueError('tournament file column header not recognized')

    # concatenate train and tournament data
    ids = np.concatenate((ids, ids2))
    era = np.concatenate((era, era2))
    region = np.concatenate((region, region2))
    x = np.concatenate((x, x2))
    y = np.concatenate((y, y2))

    return Data(ids, era, region, x, y)


def load_csv(file_like):

    # load data as np.array
    a = pd.read_csv(file_like, header=0)
    header = a.columns.values
    a = a.values

    # convert arrays from object dtype
    ids = a[:, 0].astype(str)
    era = a[:, 1].astype(str)
    region = a[:, 2].astype(str)
    x = a[:, 3:-1].astype(np.float64)
    y = a[:, -1]
    y[y == ''] = np.nan
    y = y.astype(np.float64)

    return header, ids, era, region, x, y


def expected_headers():
    header = ['id', 'era', 'data_type']
    header += ['feature'+str(i) for i in range(1, 51)]
    header += ['target']
    return np.array(header)
