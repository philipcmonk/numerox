import zipfile

import numpy as np
import pandas as pd

TRAIN_FILE = 'numerai_training_data.csv'
TOURNAMENT_FILE = 'numerai_tournament_data.csv'
HDF_KEY = 'numerox_data'
TOURNAMENT_REGIONS = ['validation', 'test', 'live']


class Data(object):

    def __init__(self, df):
        self.df = df

    @property
    def ids(self):
        "Return ids as a numpy str array"
        return self.df.index.values.astype(str)

    @property
    def x(self):
        "Return features, x, as a numpy array"
        names = self._x_names()
        return self.df[names].values

    def _x_names(self):
        "Return list of column names of features, x, in dataframe"
        cols = self._column_list()
        names = [n for n in cols if n.startswith('x')]
        if len(names) == 0:
            raise IndexError("Could not find any features (x)")
        return names

    @property
    def y(self):
        "Return y as a 1d numpy array"
        return self.df['y'].values

    @property
    def era(self):
        "Return era as a 1d numpy str array"
        return self.df['era'].values.astype(str)

    @property
    def region(self):
        "Return region as a 1d numpy str array"
        return self.df['region'].values.astype(str)

    def __getitem__(self, index):
        "Data indexing"
        typidx = type(index)
        if isinstance(index, str):
            if index.startswith('era'):
                if len(index) < 4:
                    raise IndexError('length of era string index too short')
                return self.era_isin([index])
            else:
                if index in ('train', 'validation', 'test', 'live'):
                    return self.region_isin([index])
                elif index == 'tournament':
                    return self.region_isin(TOURNAMENT_REGIONS)
                else:
                    raise IndexError('string index not recognized')
        elif typidx is pd.Series or typidx is np.ndarray:
            idx = index
            return Data(self.df[idx])
        else:
            raise IndexError('indexing type not recognized')

    def era_isin(self, eras):
        "Copy of data that contrain only the eras in the iterable `eras`"
        idx = self.df.era.isin(eras)
        return self[idx]

    def region_isin(self, regions):
        "Copy of data that contrain only the regions in the iterable `regions`"
        idx = self.df.region.isin(regions)
        return self[idx]

    def copy(self):
        "Copy of data"
        return Data(self.df.copy())

    def save(self, path_or_buf, compress=False):
        "Save data as an hdf archive"
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
        idx = ~idx
        if idx.sum() > 0:
            mean = y[idx].mean()
        else:
            # avoid numpy warning "Mean of empty slice"
            mean = np.nan
        stats = 'mean {:.6f}, fraction missing {:.4f}'
        stats = stats.format(mean, frac)
        t.append(fmt.format('y', stats))

        return '\n'.join(t)


def load(dataset_path):
    "Load numerai dataset from hdf archive; return Data"
    df = pd.read_hdf(dataset_path)
    return Data(df)


def load_zip(dataset_path):
    "Load numerai dataset from zip archive; return Data"
    zf = zipfile.ZipFile(dataset_path)
    train = pd.read_csv(zf.open(TRAIN_FILE), header=0, index_col=0)
    tourn = pd.read_csv(zf.open(TOURNAMENT_FILE), header=0, index_col=0)
    df = pd.concat([train, tourn], axis=0)
    rename_map = {'data_type': 'region', 'target': 'y'}
    for i in range(1, 51):
        rename_map['feature' + str(i)] = 'x' + str(i)
    df.rename(columns=rename_map, inplace=True)
    return Data(df)
