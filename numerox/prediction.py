

class Prediction(object):

    def __init__(self, data, yhat):
        df = data.df.copy()
        x_cols = data._x_names()
        df.drop(x_cols, axis=1, inplace=True)
        df.insert(df.shape[1], 'yhat', yhat, allow_duplicates=False)
        self.df = df

    def __repr__(self):
        if self.df is None:
            return ''
        t = []
        fmt = '{:<10}{:>13.6f}'
        y = self.df.yhat
        t.append(fmt.format('mean', y.mean()))
        t.append(fmt.format('std', y.std()))
        t.append(fmt.format('min', y.min()))
        t.append(fmt.format('max', y.max()))
        t.append(fmt.format('rows', len(self.df)))
        t.append(fmt.format('nulls', y.isnull().sum()))
        return '\n'.join(t)
