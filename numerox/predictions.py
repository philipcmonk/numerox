import os
import glob

import pandas as pd


class Predictions(object):

    def __init__(self, df=None):
        self.df = df

    def add_prediction(self, prediction, name):
        "Add prediction with corresponding model name `name`"
        df = prediction.df.rename(columns={'yhat': name}, inplace=False)
        if self.df is None:
            pass
        else:
            # no check is currently made for overlap in yhat
            df = pd.merge(self.df, df, how='outer')
        self.df = df

    @property
    def model_names(self):
        "A list of model names currently in Predictions"
        names = self.df.columns.tolist()
        names = [n for n in names if n not in ('era', 'region', 'y')]
        return names

    def __repr__(self):
        if self.df is None:
            return ''
        t = []
        for name in self.model_names:
            t.append(name)
        return '\n'.join(t)


def load_predictions(prediction_dir, extension='.pred'):
    "Load Prediction objects (hdf) in `prediction_dir`; return Report object"
    original_dir = os.getcwd()
    os.chdir(prediction_dir)
    dfs = []
    try:
        for filename in glob.glob("*{}".format(extension)):
            prediction = None  # load_prediction(filename)
            df = prediction.df
            model = filename[:-len(extension)]
            df.rename(columns={'y': model}, inplace=True)
            dfs.append(df)
    finally:
        os.chdir(original_dir)
    df = pd.concat(dfs, axis=1, verify_integrity=True, copy=False)
    return Predictions(df)
