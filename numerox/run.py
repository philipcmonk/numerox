from numerox import Prediction, tournament_splitter, cv_splitter


def run(model, data, splitter, verbosity=2):
    prediction = Prediction()
    for data_fit, data_predict in splitter:
        yhat = model.fit_predict(data_fit, data_predict)
        prediction.append(data_predict.ids, yhat)
    return prediction


def production(model, data):
    splitter = tournament_splitter(data)
    prediction = run(model, data, splitter)
    return prediction


def backtest(model, data, kfold=5, seed=0):
    splitter = cv_splitter(data, kfold=kfold, seed=seed)
    prediction = run(model, data, splitter)
    return prediction
