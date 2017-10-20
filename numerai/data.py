import zipfile

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

TRAIN_FILE = 'numerai_training_data.csv'
TOURN_FILE = 'numerai_tournament_data.csv'
HDF_KEY = 'data_object'


class Data(object):

    def __init__(self, df):
        self.df = df

    @property
    def ids(self):
        "Return ids as a numpy str array"
        return self.df.index.values.astype(str)

    @property
    def x(self):
        "Return features as a numpy array"
        names = self._x_names()
        return self.df[names].values

    @property
    def x_df(self):
        "Return features as a pandas dataframe"
        names = self._x_names()
        return self.df[names]

    def _x_names(self):
        "Return list of column names of features, x, in dataframe"
        cols = self._column_list()
        names = [n for n in cols if n.startswith('feature')]
        if len(names) == 0:
            raise IndexError("Could not find any features (x)")
        return names

    @property
    def y(self):
        "Return targets as a 1d numpy array"
        return self.df['target'].values

    @property
    def y_dh(self):
        "Return targets as a pandas dataframe"
        return self.df['target']

    @property
    def era(self):
        "Return era as a 1d numpy str array"
        return self.df['era'].values.astype(str)

    @property
    def era_dh(self):
        "Return era as a pandas dataframe"
        return self.df['era']

    @property
    def region(self):
        "Return region as a 1d numpy str array"
        return self.df['data_type'].values.astype(str)

    @property
    def region_dh(self):
        "Return region as a pandas dataframe"
        return self.df['data_type']

    def __getitem__(self, index):
        "Index into a data object. Go ahead, I dare you."
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
        elif typidx is pd.Series or typidx is np.ndarray:
            idx = index
        else:
            raise IndexError('indexing type not recognized')
        d = Data(self.df[idx])
        return d

    def cv(self, kfold=5, random_state=None):
        "Cross validation iterator that yields train, test data across eras"
        kf = KFold(n_splits=kfold, shuffle=True, random_state=random_state)
        eras = self.era_dh.unique()
        for train_index, test_index in kf.split(eras):
            idx = self.df.era.isin(eras[train_index])
            dtrain = self[idx]
            idx = self.df.era.isin(eras[test_index])
            dtest = self[idx]
            yield dtrain, dtest

    def to_hdf(self, path_or_buf, **kwargs):
        "Save data object as a hdf archive"
        self.df.to_hdf(path_or_buf, HDF_KEY, **kwargs)

    def _column_list(self):
        "Return column names of dataframe as a list"
        return self.df.columns.values.tolist()

    @property
    def size(self):
        return self.df.size

    @property
    def shape(self):
        return self.df.shape

    def __len__(self):
        return self.df.__len__()


def load_hdf(dataset_path):
    "Load numerai dataset from hdf archive; return Data"
    df = pd.read_hdf(dataset_path)
    return Data(df)


def load_zip(dataset_path):
    "Load numerai dataset from zip archive; return Data"
    zf = zipfile.ZipFile(dataset_path)
    train = pd.read_csv(zf.open(TRAIN_FILE), header=0, index_col=0)
    tourn = pd.read_csv(zf.open(TOURN_FILE), header=0, index_col=0)
    df = pd.concat([train, tourn], axis=0)
    return Data(df)
