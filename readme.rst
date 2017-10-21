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
includes the Data class. All examples below run. Numerox does not yet include
unit tests (yikes!) so it is too early to mortgage your house for stake money.

Load data quickly
=================

Is slow loading of Numerai zip files getting in the way of your overfitting?
Use HDF!

Load the dataset from a Numerai zip archive::

    import numerox as nx
    data = nx.load_zip('numerai_dataset.zip')

Save data object to HDF::

    data.to_hdf('numerai_dataset.h5')

Just think how quickly you will overfit the data::

    In [1]: timeit nx.load_zip('numerai_dataset.zip')
    1 loop, best of 3: 7.31 s per loop
    In [2]: timeit nx.load_hdf('numerai_dataset.h5')
    1 loop, best of 3: 174 ms per loop

Indexing
========

Which era do you want to overfit::

    In [3]: len(data['era92'])
    Out[3]: 6048
    In [4]: len(data['eraX'])
    Out[4]: 274967

Here's where the money is::

    In [5]: data['live'].x
    Out[5]:
    array([[ 0.44528,  0.54614,  0.7495 , ...,  0.72822,  0.38965,  0.45422],
           [ 0.39358,  0.59676,  0.54018, ...,  0.54839,  0.45487,  0.3092 ],
           [ 0.41948,  0.21274,  0.41617, ...,  0.56515,  0.49289,  0.64172],
           ...,
           [ 0.58081,  0.56891,  0.48076, ...,  0.69221,  0.48768,  0.44878],
           [ 0.41867,  0.54406,  0.53561, ...,  0.47159,  0.49141,  0.49558],
           [ 0.53379,  0.39891,  0.50305, ...,  0.4391 ,  0.40262,  0.48789]])

Besides strings, you can also index with numpy arrays or pandas series.

You can pull out numpy arrays like so ``data.x``, ``data.y``, ``data.era``,
and pandas dataframes like so ``data.x_df``, ``data.y_df``, ``data.era_df``.


Cross validation
================

To make your overfitting modestly challenging use cross validation::

    In [6]: for dtrain, dtest in nx.cv(data['train'], kfold=5, random_state=0):
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
- pytables (fast archiving)
- sklearn

Install with pipi (not yet working)::

    $ sudo pip install numerox

Resources
=========

Questions, comments, suggests, money: Numerai's slack channel and on github:
https://github.com/kwgoodman/numerox

License
=======

Numerox is distributed under the GPL v3+. See the LICENSE file for details.
