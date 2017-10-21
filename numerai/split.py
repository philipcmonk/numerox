from sklearn.model_selection import KFold


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
