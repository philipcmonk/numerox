from sklearn.linear_model import LogisticRegression


class LogReg(object):  # must have fit and predict methods

    def __init__(self, C):  # use whatever input parameters you need
        self.C = C
        self.model = None

    def fit(self, data):  # data must be the only input parameter
        self.model = LogisticRegression(C=self.C)
        self.model.fit(data.x, data.y)

    def predict(self, data):  # data must be the only input parameter
        if self.model is None:
            raise ValueError("model has not been fit")
        yhat = self.model.predict_prob(data.x)[:, 1]
        return data.ids, yhat
