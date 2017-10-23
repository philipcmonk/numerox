import zipfile

import numpy as np
import pandas as pd

TRAIN_FILE = 'numerai_training_data.csv'
TOURN_FILE = 'numerai_tournament_data.csv'
HDF_KEY = 'numerox_data'


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
    def era(self):
        "Return era as a 1d numpy str array"
        return self.df['era'].values.astype(str)

    @property
    def region(self):
        "Return region as a 1d numpy str array"
        return self.df['data_type'].values.astype(str)

    def __getitem__(self, index):
        "Index into a data object. Go ahead, I dare you."
        typidx = type(index)
        if isinstance(index, str):
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

    def copy(self):
        "Copy of data"
        return Data(self.df.copy())

    def to_hdf(self, path_or_buf, compress=False):
        "Save data object as a hdf archive"
        if compress:
            self.df.to_hdf(path_or_buf, HDF_KEY, complib='zlib', complevel=4)
        else:
            self.df.to_hdf(path_or_buf, HDF_KEY)

    def _column_list(self):
        "Return column names of dataframe as a list"
        return self.df.columns.values.tolist()

    def unique_era(self):
        "array of unique eras"
        return np.unique(self.era)

    def unique_region(self):
        "array of unique regions"
        return np.unique(self.region)

    @property
    def size(self):
        return self.df.size

    @property
    def shape(self):
        return self.df.shape

    def __len__(self):
        return self.df.__len__()

    def __repr__(self):

        t = []
        fmt = '{:<10}{:<}'

        # region
        r = self.unique_region()
        stats = ', '.join(r)
        t.append(fmt.format('region', stats))

        # ids
        t.append(fmt.format('rows', len(self)))

        # era
        e = self.unique_era()
        stats = '{}, {} - {}'.format(e.size, e[0], e[-1])
        t.append(fmt.format('era', stats))

        # x
        x = self.x
        stats = '{}, min {:.4f}, mean {:.4f}, max {:.4f}'
        stats = stats.format(x.shape[1], x.min(), x.mean(), x.max())
        t.append(fmt.format('x', stats))

        # y
        y = self.y
        idx = np.isnan(y)
        frac = idx.mean()
        if idx.sum() > 0:
            mean = y[~idx].mean()
        else:
            # avoid numpy warning "Mean of empty slice"
            mean = np.nan
        stats = 'mean {:.6f}, fraction missing {:.4f}'
        stats = stats.format(mean, frac)
        t.append(fmt.format('y', stats))

        return '\n'.join(t)


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
