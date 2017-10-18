numerai
=======

The example code that is distributed with the numerai dataset contains the
lines::

    training_data = pd.read_csv('numerai_training_data.csv', header=0)
    prediction_data = pd.read_csv('numerai_tournament_data.csv', header=0)

    features = [f for f in list(training_data) if "feature" in f]
    X = training_data[features]
    Y = training_data["target"]
    x_prediction = prediction_data[features]
    ids = prediction_data["id"]

    model.fit(X, Y)
    y_prediction = model.predict(x_prediction)

This method of pulling out the numerai data from the pandas dataframe is
reproduced thoughout the numerai code base. Which suggest that we should
create a numerai Data object that takes care of the details::

    data = ni.load('numerai_dataset.zip')
    train = data['train']
    predict = data['tournament']

    model.fit(train.x, train.y)
    yhat = model.predict(predict.x)

(Loading from the zip archive is slow. So I would have a function that creates
an hdf5 archive.)

And why not add a method that handles cross validation across the eras::

    for train, predict in data['train'].cv(kfold=5, random_state=0):
        model.fit(train.x, train.y)
        y = model.predict(predict.x)

Some of the submissions to numerai do not pass consistency and concordance.
Although these functions are now open source, they are not yet accessible to
average users. So the package that contains the Data class could also contain
the consistency, concordance, and originality functions.

The package that I am proposing to write would let the user create a model
that contains fit and predict methods (examples would be provided) and then
the package would run a backtest (CV) across the training data and report the
results (logloss, consistency, AUC, etc). Once they are happy with their model,
there is a production function that downloads the data, runs your model and
uploads the results.

If we wish to fully build out the package I would add unit tests and CI with
Travis (and maybe Appveyor). And I would upload it to pypi so that users could
do::

    $ pip install numerai

The CI would make it much easier to review PRs. And unit tests would make
refactoring much easier.

This is a package that I believe would be used by many, including internally
at Numerai and in the packages you have already open sourced.

If you are interested in having me start working on it (even just having a
data class would be useful, so that reduces the risk of a full build out),
let me know and we can discuss compensation.

Indexing
========

Here is an indexing demo of the working prototype in ``data.py``::

    In [1]: import numerai as ni
    In [2]: data = ni.load_zip('numerai_dataset_20171017.zip')

    In [3]: data.size
    Out[3]: 884545

    # str indexing
    In [4]: data['era92'].size
    Out[4]: 6048
    In [5]: data['eraX'].size
    Out[5]: 274967
    In [6]: data['tournament'].size
    Out[6]: 348832
    In [7]: data['live'].size
    Out[7]: 6804
    In [8]: data['live'].x.shape
    Out[8]: (6804, 50)

    # ndarray indexing
    In [9]: data[data.x[:, 0] > 0.5].size
    Out[9]: 347176
    In [10]: data[data.x[:, 0] <= 0.5].size
    Out[10]: 537369

Cross validation
================

Here is a cross validation demo of the working prototype in ``data.py``::

    In [1]: import numerai as ni
    In [2]: data = ni.load_zip('numerai_dataset_20171017.zip')

    In [3]: for dtrain, dtest in data['train'].cv(kfold=5, random_state=0):
       ...:     print dtrain.size, dtest.size
       ...:
    428887 106826
    429062 106651
    428111 107602
    428225 107488
    428567 107146
