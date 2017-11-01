import pprint

from numerox import Prediction, tournament_splitter, cv_splitter


def run(model, data, splitter, verbosity=2):
    if verbosity > 0:
        pprint.pprint(model)
    prediction = Prediction()
    for data_fit, data_predict in splitter:
        ids, yhat = model.fit_predict(data_fit, data_predict)
        prediction.append(ids, yhat)
        if verbosity > 1:
            prediction.performance(data)
    if verbosity == 1:
        prediction.performance(data)
    return prediction


def production(model, data):
    splitter = tournament_splitter(data)
    prediction = run(model, data, splitter)
    return prediction


def backtest(model, data, kfold=5, seed=0, verbosity=2):
    splitter = cv_splitter(data, kfold=kfold, seed=seed)
    prediction = run(model, data, splitter, verbosity)
    return prediction
