numerox
=======

Numerox is a Numerai machine learning competition toolbox written in Python.

All you have to do is create a model. (Look away while I type my prize-winning
model)::

    from sklearn.linear_model import LogisticRegression

    class LogRegModel(object):  # must have fit_predict method

        def __init__(self, C):  # add whatever inputs you need
            self.C = C

        # must take two datas (fit, predict) and return (ids, yhat) arrays
        def fit_predict(self, data_fit, data_predict):
            model = LogisticRegression(C=self.C)
            model.fit(data_fit.x, data_fit.y)
            yhat = model.predict_proba(data_predict.x)[:, 1]
            return data_predict.ids, yhat

Once you have a model numerox will do the rest::

    >>> model = MyModel(C=1)
    >>> data = nx.load_data('numerai_dataset.hdf')
    >>> prediction = nx.backtest(model, data['train'], kfold=5, seed=0, verbosity=1)
          logloss   auc     acc     ystd
    mean  0.692770  0.5197  0.5137  0.0281  |  region   train
    std   0.003196  0.0314  0.0231  0.0019  |  eras     85
    min   0.683797  0.4435  0.4545  0.0252  |  consis   0.5176
    max   0.701194  0.6027  0.5751  0.0316  |  75th     0.6944

What the hell, looks good enough. Let's make a submission file for the
tournament (we will fail to pass the consistency threshold)::

    >>> prediction = nx.production(model, data)
          logloss   auc     acc     ystd
    mean  0.692473  0.5212  0.5149  0.0270  |  region   validation
    std   0.002360  0.0250  0.0175  0.0004  |  eras     12
    min   0.687518  0.4926  0.4954  0.0262  |  consis   0.5000
    max   0.695493  0.5754  0.5555  0.0274  |  75th     0.6940
    >>> prediction.to_csv('logreg.csv')  # saves 6 decimal places by default

Warning
=======

This preview release has minimal unit tests coverage (yikes!). In the next
release I will vengefully break any code you write using numerox---the api
is not yet stable.

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

Numerox comes with a small dataset to play with::

    >>> nx.load_play_data()
    region    live, test, train, validation
    rows      8795
    era       98, [era1, eraX]
    x         50, min 0.0259, mean 0.4995, max 0.9913
    y         mean 0.502646, fraction missing 0.3126

It is about 1% of a regular Numerai dataset, so contains around 60 rows per
era.

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
