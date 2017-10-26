import pandas as pd

HDF_PREDICTION_KEY = 'numerox_prediction'


class Prediction(object):

    def __init__(self, df=None):
        self.df = df

    @property
    def ids(self):
        "Copy of ids as a numpy str array or None is empty"
        if self.df is None:
            return None
        return self.df.index.values.astype(str)

    @property
    def y(self):
        "Copy of y as a 1d numpy array or None is empty"
        if self.df is None:
            return None
        return self.df['y'].values.copy()

    def append(self, ids, y):
        df = pd.DataFrame(data={'y': y}, index=ids)
        if self.df is None:
            df.index.rename('ids', inplace=True)
        else:
            try:
                df = pd.concat([self.df, df], verify_integrity=True)
            except ValueError:
                # pandas doesn't raise expected IndexError and for our large
                # number of y, the id overlaps that it prints can be very long
                raise IndexError("Overlap in ids found")
        self.df = df

    def save(self, path_or_buf, compress=True):
        "Save prediction as an hdf archive; raises if nothing to save"
        if self.df is None:
            raise ValueError("Prediction object is empty; nothing to save")
        if compress:
            self.df.to_hdf(path_or_buf, HDF_PREDICTION_KEY,
                           complib='zlib', complevel=4)
        else:
            self.df.to_hdf(path_or_buf, HDF_PREDICTION_KEY)

    def copy(self):
        "Copy of prediction"
        if self.df is None:
            return Prediction(None)
        return Prediction(self.df.copy(deep=True))

    @property
    def size(self):
        if self.df is None:
            return 0
        return self.df.size

    @property
    def shape(self):
        if self.df is None:
            return tuple()
        return self.df.shape

    def __len__(self):
        "Number of rows"
        if self.df is None:
            return 0
        return self.df.__len__()

    def _column_list(self):
        "Return column names of dataframe as a list"
        return self.df.columns.values.tolist()

    def __repr__(self):
        if self.df is None:
            return ''
        t = []
        fmt = '{:<10}{:>13.6f}'
        y = self.df.y
        t.append(fmt.format('mean', y.mean()))
        t.append(fmt.format('std', y.std()))
        t.append(fmt.format('min', y.min()))
        t.append(fmt.format('max', y.max()))
        t.append(fmt.format('rows', len(self.df)))
        t.append(fmt.format('nulls', y.isnull().sum()))
        return '\n'.join(t)


def load_prediction(file_path):
    "Load prediction object from hdf archive; return Prediction"
    df = pd.read_hdf(file_path, key=HDF_PREDICTION_KEY)
    return Prediction(df)
