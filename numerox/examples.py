

class LogRegModel(object):  # must have fit_predict method

    def __init__(self, C):  # add whatever inputs you need
        self.C = C

    # must take two datas (train, predict) and return ids, y arrays
    def fit_predict(self, data_train, data_predict):
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(C=self.C)
        model.fit(data_train.x, data_train.y)
        yhat = model.predict_prob(data_predict.x)[:, 1]
        return data_predict.ids, yhat
