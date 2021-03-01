# this merges _sm and _med to one big data
import numpy as np

def merge():
    with open('data_sm.npy', 'rb') as fl:
        data_1 = np.load(fl)
        res_1 = np.load(fl)

    with open('data_med.npy', 'rb') as fl:
        data_2 = np.load(fl)
        res_2 = np.load(fl)

    data = np.concatenate((data_1, data_2))
    res = np.concatenate((res_1, res_2))

    with open('data_all.npy', 'wb') as fl:
        np.save(fl, data)
        np.save(fl, res)

merge()
