from sklearn.linear_model import LogisticRegression


class LogRegModel(object):  # must have fit_predict method

    def __init__(self, C):  # add whatever inputs you need
        self.C = C

    def fit_predict(self, data_train, data_predict):  # must take two datas
        model = LogisticRegression(C=self.C)
        model.fit(data_train.x, data_train.y)
        yhat = model.predict_prob(data_predict.x)[:, 1]
        return data_predict.ids, yhat  # must return ids, y arrays
