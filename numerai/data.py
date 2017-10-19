import zipfile

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


class Data(object):

    def __init__(self, df):
        self.df = df

    def cv(self, kfold=5, random_state=None):
        kf = KFold(n_splits=kfold, shuffle=True, random_state=random_state)
        # TODO following two lines are awkward
        era = self.df.era.values.astype(str)
        eras = self.df.era.unique().astype(str)
        for train_index, test_index in kf.split(eras):
            idx = np.zeros(self.size, dtype=np.bool)
            for i in train_index:
                idx = np.logical_or(idx, era == eras[i])
            dtrain = self[idx]
            idx = np.zeros(self.size, dtype=np.bool)
            for i in test_index:
                idx = np.logical_or(idx, era == eras[i])
            dtest = self[idx]
            yield dtrain, dtest

    def __getitem__(self, index):

        typidx = type(index)

        if typidx is str:
            if index.startswith('era'):
                idx = self.df.era == index
            else:
                if index == 'train':
                    idx = self.df.data_type == 'train'
                elif index == 'validation':
                    idx = self.df.data_type == 'validation'
                elif index == 'test':
                    idx = self.df.data_type == 'test'
                elif index == 'live':
                    idx = self.df.data_type == 'live'
                elif index == 'tournament':
                    idx = self.df.data_type == 'validation'
                    idx = np.logical_or(idx, self.df.data_type == 'test')
                    idx = np.logical_or(idx, self.df.data_type == 'live')
        elif typidx is np.ndarray:
            idx = index
        else:
            raise IndexError('indexing type not recognized')

        d = Data(self.df[idx])

        return d

    @property
    def size(self):
        return self.df.shape[0]


TRAIN_FILE = 'numerai_training_data.csv'
TOURN_FILE = 'numerai_tournament_data.csv'


def load_zip(dataset_zip):

    # load from numerai zip archive
    zf = zipfile.ZipFile(dataset_zip)
    train = pd.read_csv(zf.open(TRAIN_FILE), header=0, index_col=0)
    tourn = pd.read_csv(zf.open(TOURN_FILE), header=0, index_col=0)

    # check headers
    header0 = expected_headers()
    header = train.columns.values.astype(str)
    if (header != header0).any():
        raise ValueError('train file column header not recognized')
    header = tourn.columns.values.astype(str)
    if (header != header0).any():
        raise ValueError('tournament file column header not recognized')

    # concatenate train and tournament data
    df = pd.concat([train, tourn], axis=0)

    return Data(df)


def expected_headers():
    header = ['era', 'data_type']
    header += ['feature'+str(i) for i in range(1, 51)]
    header += ['target']
    return np.array(header)
