import numpy as np
from sklearn.model_selection import KFold


def tournament_splitter(data):
    "Single split yielding train, tournament data"
    yield data['train'], data['tournament']


def validation_splitter(data):
    "Single split yielding train, validation data"
    yield data['train'], data['validation']


def cheat_splitter(data):
    "Single split yielding train+validation, tournament data"
    yield data.region_isin(['train', 'validation']), data['tournament']


def split_splitter(data, fit_fraction, seed=0):
    "Single fit-predict split of train data"
    data = data['train']
    eras = data.unique_era()
    rs = np.random.RandomState(seed)
    rs.shuffle(eras)
    nfit = int(fit_fraction * eras.size + 0.5)
    data_fit = data.era_isin(eras[:nfit])
    data_predict = data.era_isin(eras[nfit:])
    yield data_fit, data_predict


def cv_splitter(data, kfold=5, seed=0):
    "K-fold cross validation fit-predict splits across train eras"
    data = data['train']
    kf = KFold(n_splits=kfold, shuffle=True, random_state=seed)
    eras = data.unique_era()
    for train_index, test_index in kf.split(eras):
        era_train = [eras[i] for i in train_index]
        era_test = [eras[i] for i in test_index]
        dtrain = data.era_isin(era_train)
        dtest = data.era_isin(era_test)
        yield dtrain, dtest
