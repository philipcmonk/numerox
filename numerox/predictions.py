import os
import glob

import pandas as pd


class Predictions(object):

    def __init__(self, df=None):
        self.df = df


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
