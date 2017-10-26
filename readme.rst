numerox
=======

Numerox is a Numerai machine learning competition toolbox written in Python.

All you have to do is create a model. (Look away while I type my prize-winning
model)::

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

Once you have a model numerox will do the rest::

    model = MyModel(C=1)
    data = nx.load_data('numerai_dataset.hdf')
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

I lied
======

The examples above are merely my plan for numerox. This preview release only
includes the Data class and has minimal unit tests coverage (yikes!). The
examples below work.

Data class
==========

You can create a data object from the zip archive provided by Numerai::

    >>> import numerox as nx
    >>> data = nx.load_zip('numerai_dataset.zip')
    >>> data
    region    live, test, train, validation
    rows      884544
    era       98, [era1, eraX]
    x         50, min 0.0000, mean 0.4993, max 1.0000
    y         mean 0.499961, fraction missing 0.3109

But that is slow (~7 seconds) which is painful for dedicated overfitters.
Let's create an HDF5 archive::

    >>> data.save('numerai_dataset.hdf')
    >>> data2 = nx.load_data('numerai_dataset.hdf')

That loads quickly (~0.2 seconds, but takes more disk space than the
unexpanded zip archive).

Data indexing is done by rows, not columns::

    >>> data[data.y == 0]
    region    train, validation
    rows      304813
    era       97, [era1, era97]
    x         50, min 0.0000, mean 0.4993, max 1.0000
    y         mean 0.000000, fraction missing 0.0000

You can also index with special strings. Here are two examples::

    >>> data['era92']
    region    validation
    rows      6048
    era       1, [era92, era92]
    x         50, min 0.0308, mean 0.4993, max 1.0000
    y         mean 0.500000, fraction missing 0.0000

    >>> data['tournament']
    region    live, test, validation
    rows      348831
    era       13, [era86, eraX]
    x         50, min 0.0000, mean 0.4992, max 1.0000
    y         mean 0.499966, fraction missing 0.7882

If you wish to extract more than one era (I hate these eras)::

    >>> data.era_isin(['era92', 'era93'])
    region    validation
    rows      12086
    era       2, [era92, era93]
    x         50, min 0.0177, mean 0.4993, max 1.0000
    y         mean 0.500000, fraction missing 0.0000

You can do the same with regions::

    >>> data.region_isin(['test', 'live'])
    region    live, test
    rows      274966
    era       1, [eraX, eraX]
    x         50, min 0.0000, mean 0.4992, max 1.0000
    y         mean nan, fraction missing 1.0000

Or you can remove regions (or eras)::

    >>> data.region_isnotin(['test', 'live'])
    region    train, validation
    rows      609578
    era       97, [era1, era97]
    x         50, min 0.0000, mean 0.4993, max 1.0000
    y         mean 0.499961, fraction missing 0.0000

You can concatenate data objects (as long as the ids don't overlap) by
adding them together. Let's add validation era92 to the training data::

    >>> data['train'] + data['era92']
    region    train, validation
    rows      541761
    era       86, [era1, era92]
    x         50, min 0.0000, mean 0.4993, max 1.0000
    y         mean 0.499960, fraction missing 0.0000

Or, let's go crazy::

    >>> nx.concat([data['live'], data['era1'], data['era92']])
    region    live, train, validation
    rows      19194
    era       3, [era1, eraX]
    x         50, min 0.0000, mean 0.4992, max 1.0000
    y         mean 0.499960, fraction missing 0.3544

You can pull out numpy arrays (copies, not views) like so ``data.ids``,
``data.era``, ``data.region``, ``data.x``, ``data.y``.

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

After you have installed numerox, run the unit tests (please report any
failures)::

    >>> import numerox as nx
    >>> nx.test()
    <snip>
    Ran 4 tests 0.129
    OK
    <nose.result.TextTestResult run=4 errors=0 failures=0>

Resources
=========

Questions, comments, suggests: Numerai's slack channel and on github:
https://github.com/kwgoodman/numerox/issues.

License
=======

Numerox is distributed under the Simplified BSD. See LICENSE file for details.
