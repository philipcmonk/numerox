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

We can also index into a particular era::

    data['era92']

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

And of course there is all sorts of fun things we could add, like a user
report::

    $ ni.user_report('bps')
    bps, 23 rounds
           0.86957  consistency
           0.69565  super consistency
           0.58394  logloss dominance
          65.00000  best rank
           0.53713  logloss correlation
           0.69289  logloss mean
           0.00048  logloss std
           0.13043  live < validation
          50.29000  usd main
        1825.58000  usd stake
          52.79000  nmr main
          50.00000  nmr burn
        1915.37305  earnings at $14.16/NMR

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

My bottleneck packages is fully unit tested, uses CI with Travis and Appveyor
and is packaged by many Linux distributions.
