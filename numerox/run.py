from numerox import Prediction


def run(model, data, splitter, prediction=None, verbosity=2):
    if prediction is None:
        prediction = Prediction()
    for data_fit, data_predict in splitter:
        yhat = model.fit_predict(data_fit, data_predict)
        prediction.append(data_predict.ids, yhat)
    return prediction
