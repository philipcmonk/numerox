from sklearn.linear_model import LogisticRegression
from numerox.core import api


class LogReg(api.Model):

    def __init__(self, C):
        # init is not part of api.Model so do whatever you want
        self.C = C
        self.model = None

    def fit(self, data, seed):
        # seed is not used in this model but is part of api.Model
        self.model = LogisticRegression(C=self.C)
        self.model.fit(data.x, data.y)

    def predict(self, data):
        if self.model is None:
            raise ValueError("model has not been fit")
        yhat = self.model.predict_prob(data.x)[:, 1]
        return data.ids, yhat
