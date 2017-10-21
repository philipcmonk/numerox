numbox
======

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

    data = nb.load('numerai_dataset.zip')
    train = data['train']
    predict = data['tournament']

    model.fit(train.x, train.y)
    yhat = model.predict(predict.x)

The package that I am proposing to write would let the user create a model
that contains fit and predict methods (examples would be provided) and then
the package would run a backtest (CV) across the training data and report the
results (logloss, consistency, AUC, etc). Once they are happy with their model,
there is a production function that downloads the data, runs your model and
uploads the results.

This package would lower the barrier to entry for new users and would improve
entries by reducing bugs and making sure users properly use cross validation
and/or splits across eras instead of rows.

If we wish to fully build out the package I would add unit tests and CI with
Travis (and maybe Appveyor). And I would upload it to pypi so that users could
do::

    $ pip install numbox

The CI would make it much easier to review PRs. And unit tests would make
refactoring much easier.

This is a package that I believe would be used by many, including internally
at Numerai and in the packages you have already open sourced.

If you are interested in having me start working on it (even just having a
data class would be useful, so that reduces the risk of a full build out),
let me know and we can discuss funding.

Indexing
========

Which era do you want to overfit::

    In [1]: import numbox as nb
    In [2]: data = nb.load_zip('numerai_dataset_20171017.zip')

    In [3]: data['era92'].size
    Out[3]: 6048
    In [4]: data['eraX'].size
    Out[4]: 274967

Here's where the money is::

    In [5]: data['live'].x.shape
    Out[5]: (6804, 50)

But you will first have to upload predictions for::

    In [6]: data['tournament'].size
    Out[6]: 348832

Besides strings, you can also index with numpy arrays or pandas series::

    In [7]: data[data.x[:, 0] > 0.5].size
    Out[7]: 347176
    In [8]: data[data.x[:, 0] <= 0.5].size
    Out[8]: 537369

Cross validation
================

To make your overfitting modestly challenging use cross validation::

    In [9]: for dtrain, dtest in nb.cv(data['train'], kfold=5, random_state=0):
       ...:     print len(dtrain), len(dtest)
       ...:
    428333 107380
    428841 106872
    428195 107518
    428218 107495
    429265 106448

Loading data quickly
====================

Is slow loading of data from a Numerai zip file getting in the way of your
overfitting? Save it to HDF!

Load the dataset from a Numerai zip archive::

    In [1]: data = nb.load_zip('numerai_dataset.zip')

Save the data object to HDF::

    In [2]: data.to_hdf('numerai_dataset.h5')

Just think how quickly you will overfit the data::
    
    In [3]: timeit nb.load_zip('numerai_dataset.zip')
    1 loop, best of 3: 7.31 s per loop
    In [4]: timeit nb.load_hdf('numerai_dataset.h5')
    1 loop, best of 3: 174 ms per loop

Package name
============

Naming the package numerai is problematic. I think of it as a toolbox. So
how about using the name numbox as in: Numbox is a Numerai machine learning
competition toolbox.
