from sklearn.linear_model import LogisticRegression


class LogRegModel(object):  # must have fit_predict method

    def __init__(self, C):  # add whatever inputs you need
        self.C = C

    # must take two datas (fit, predict) and return (ids, yhat) arrays
    def fit_predict(self, data_fit, data_predict):
        model = LogisticRegression(C=self.C)
        model.fit(data_fit.x, data_fit.y)
        yhat = model.predict_prob(data_predict.x)[:, 1]
        return data_predict.ids, yhat
