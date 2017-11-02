import os
import glob

import pandas as pd

from numerox.prediction import load_prediction


class Report(object):

    def __init__(self, df=None):
        self.df = df

    def performance(self):
        pass


def load_report(data, prediction_dir, extension='.pred'):
    "Load Prediction objects (hdf) in `prediction_dir`; return Report object"

    # load prediction objects
    original_dir = os.getcwd()
    os.chdir(prediction_dir)
    dfs = []
    try:
        for filename in glob.glob("*{}".format(extension)):
            prediction = load_prediction(filename)
            df = prediction.df
            model = filename[:-len(extension)]
            df.rename(columns={'yhat': model}, inplace=True)
            dfs.append(df)
    finally:
        os.chdir(original_dir)

    # concatenate predictions
    df = pd.concat(dfs, axis=1, verify_integrity=True, copy=False)

    # add in era, region, and y info
    ery = data.df[['era', 'region', 'y']]
    df = pd.merge(ery, df, left_index=True, right_index=True, how='right')

    return Report(df)
