import numpy as np

from numerox import Data


def shares_memory(data1, data_or_array2):
    "True if `data1` shares memory with `data_or_array2`; False otherwise"
    isdata = isinstance(data_or_array2, Data)
    for col in data1._column_list():
        a1 = data1.df[col].values
        if isdata:
            a2 = data_or_array2.df[col].values
        else:
            a2 = data_or_array2
        if np.shares_memory(a1, a2):
            return True
    return False
