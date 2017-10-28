from sklearn.model_selection import KFold


def cv_splitter(data, kfold=5, seed=0):
    "Cross validation iterator that yields train, test data across eras"
    kf = KFold(n_splits=kfold, shuffle=True, random_state=seed)
    eras = data.unique_era()
    for train_index, test_index in kf.split(eras):
        era_train = [eras[i] for i in train_index]
        era_test = [eras[i] for i in test_index]
        dtrain = data.era_isin(era_train)
        dtest = data.era_isin(era_test)
        yield dtrain, dtest
