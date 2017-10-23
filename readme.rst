numerox
=======

Numerox is a Numerai machine learning competition toolbox written in Python.

All you have to do is create a model. (Look away while I type my prize-winning
model)::

    from sklearn.linear_model import LogisticRegression
    from numerox.core import api

    class MyModel(api.Model):

        def __init__(self, C):
            # init is not part of api.Model so do whatever you want
            self.C = C
            self.model = None

        def fit(self, data, seed):
            # seed is not used in this model but is part of the Model api
            self.model = LogisticRegression(C=self.C)
            self.model.fit(data.x, data.y)

        def predict(self, data):
            if self.model is None:
                raise ValueError("model has not been fit")
            yhat = self.model.predict_prob(data.x)[:, 1]
            return data.ids, yhat

(api.Model is an abstract base class that defines and enforces the interface
that numerox expects from your model.)

Once you have a model numerox will do the rest::

    model = MyModel(C=1)
    data = nx.load_zip('numerai_dataset.zip')
    # pick a random seed that equals hoped for USD prize money
    prediction = nx.backtest(model, data, kfold=5, seed=1000)

After each fold you'll see a cumulative report::

    logloss    85  acc     auc     ymean   ystd    ymax   num    minutes
    0.692463  mea  0.5153  0.5219  0.5000  0.0130  0.039  6303   0.03
    0.001428  std  0.0232  0.0314  0.0000  0.0000  0.001   236   0.00
    0.689814  min  0.4652  0.4505  0.5000  0.0130  0.038  5927   0.03
    0.695774  max  0.5574  0.5809  0.5000  0.0130  0.040  6793   0.03
              con  0.7059  75 pct  0.69333 sharpe  0.479

What the hell, looks good enough. Let's make a submission file for the
tournament::

    # It is very bad luck to pick a random seed that equals hoped for prize
    prediction = nx.production(model, data, seed=0)
    prediction.to_csv('mymodel.csv')

If you want to look at the results in more detail, make a report::

    report = nx.Report(data, prediction)
    report.auc()
    ...

I lied
======

The examples above are merely my plan for numerox. This preview release only
includes the Data class and has minimal unit tests coverage (yikes!).

Load data quickly
=================

Is slow loading of Numerai zip files getting in the way of your overfitting?
Use HDF!

Load the dataset from a Numerai zip archive::

    >>> import numerox as nx
    >>> data = nx.load_zip('numerai_dataset.zip')
    >>> data
    region    live, test, train, validation
    rows      884545
    era       98, era1 - eraX
    x         50, min 0.0000, mean 0.4993, max 1.0000
    y         mean 0.499961, fraction missing 0.3109

Save data object to HDF::

    >>> data.save('numerai_dataset.hdf')

Just think how quickly you will overfit the data::

    >>> timeit nx.load('numerai_dataset.hdf')
    1 loop, best of 3: 174 ms per loop
    >>> timeit nx.load_zip('numerai_dataset.zip')
    1 loop, best of 3: 7.31 s per loop

Indexing
========

Which era do you want to overfit::

    >>> data['era92']
    region    validation
    rows      6048
    era       1, era92 - era92
    x         50, min 0.0308, mean 0.4993, max 1.0000
    y         mean 0.500000, fraction missing 0.0000

    >>> data['eraX']
    region    live, test
    rows      274967
    era       1, eraX - eraX
    x         50, min 0.0000, mean 0.4992, max 1.0000
    y         mean nan, fraction missing 1.0000

Here's where the money is::

    >>> data['live']
    region    live
    rows      6804
    era       1, eraX - eraX
    x         50, min 0.0348, mean 0.4993, max 0.9897
    y         mean nan, fraction missing 1.0000

Besides strings, you can also index with numpy arrays or pandas series.

You can pull out numpy arrays like so ``data.x``, ``data.y``, ``data.era``,
etc.


Cross validation
================

To make your overfitting modestly challenging use cross validation::

    >>> for dtrain, dtest in nx.cv(data['train'], kfold=5, random_state=0):
       ...:     print len(dtrain), len(dtest)
       ...:
    428333 107380
    428841 106872
    428195 107518
    428218 107495
    429265 106448

Install
=======

This is what you need to run numerox::

- python
- numpy
- pandas
- pytables
- sklearn
- nose (unit tests)

Install with pipi (not yet working)::

    $ sudo pip install numerox

After you have installed numerox, run the suite of unit tests::

    >>> import numerox as nx
    >>> nx.test()
    <snip>
    Ran 1 tests 0.044s
    OK
    <nose.result.TextTestResult run=1 errors=0 failures=0>

Please report any unit test errors or failures.


Resources
=========

Questions, comments, suggests, money: Numerai's slack channel and on github:
https://github.com/kwgoodman/numerox.

If you know of a company that might benefit from this package, then please
encourage them to fund the development (me) of numerox.

License
=======

Numerox is distributed under the GPL v3+. See the LICENSE file for details.
