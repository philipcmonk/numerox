from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier as ETC


class logistic(object):  # must have fit_predict method

    def __init__(self, C=0.00001):  # add whatever inputs you need
        self.C = C

    # must take two datas (fit, predict) and return (ids, yhat) arrays
    def fit_predict(self, data_fit, data_predict):
        model = LogisticRegression(C=self.C)
        model.fit(data_fit.x, data_fit.y)
        yhat = model.predict_proba(data_predict.x)[:, 1]
        return data_predict.ids, yhat


class extratrees(object):

    def __init__(self, ntrees=100, depth=3, nfeatures=7, seed=0):
        self.ntrees = ntrees
        self.depth = depth
        self.nfeatures = nfeatures
        self.seed = seed

    def fit_predict(self, data_fit, data_predict):
        clf = ETC(criterion='gini',
                  max_features=self.nfeatures,
                  max_depth=self.depth,
                  n_estimators=self.ntrees,
                  random_state=self.seed)
        clf.fit(data_fit.x, data_fit.y)
        yhat = clf.predict_proba(data_predict.x)[:, 1]
        return data_predict.ids, yhat
