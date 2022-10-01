# useful functions
import numpy as np
import pandas as pd
from scipy.signal import iirfilter,sosfilt, zpk2sos

def bandpass(data, freqmin, freqmax, fs, corners=4, zerophase=False):
    """
    Butterworth-Bandpass Filter.
    Straight from obspy.

    Filter data from ``freqmin`` to ``freqmax`` using ``corners``
    corners.
    The filter uses :func:`scipy.signal.iirfilter` (for design)
    and :func:`scipy.signal.sosfilt` (for applying the filter).

    :type data: numpy.ndarray
    :param data: Data to filter.
    :param freqmin: Pass band low corner frequency.
    :param freqmax: Pass band high corner frequency.
    :param df: Sampling rate in Hz.
    :param corners: Filter corners / order.
    :param zerophase: If True, apply filter once forwards and once backwards.
        This results in twice the filter order but zero phase shift in
        the resulting filtered trace.
    :return: Filtered data.
    """
    fe = 0.5 * fs
    low = freqmin / fe
    high = freqmax / fe

    if low > 1:
        msg = "Selected low corner frequency is above Nyquist."
        raise ValueError(msg)
    z, p, k = iirfilter(corners, [low, high], btype='band',
                        ftype='butter', output='zpk')
    sos = zpk2sos(z, p, k)
    if zerophase:
        firstpass = sosfilt(sos, data)
        return sosfilt(sos, firstpass[::-1])[::-1]
    else:
        return sosfilt(sos, data)


class Dataset:

    def __init__(self, path):
        self.catalog = pd.read_pickle(path)

        self.rock_exp_nums = list(set(self.catalog[self.catalog.substrate=="rock"]["expname"]))
        self.till_exp_nums = list(set(self.catalog[self.catalog.substrate=="till"]["expname"]))

    def train_test_split_by_experiment(self, test_n_exp=2):
        """Divide data up into training and test sets, where
        the test set is made up of all waveforms from `test_n_exp` experiments
        """
        rock_test_exps = np.random.choice(self.rock_exp_nums, size=test_n_exp, replace=False).tolist()
        till_test_exps = np.random.choice(self.till_exp_nums, size=test_n_exp, replace=False).tolist()
        test_exps = rock_test_exps + till_test_exps

        self.test_cat = self.catalog[self.catalog["expname"].isin(test_exps)]
        self.train_cat = self.catalog[~self.catalog["expname"].isin(test_exps)]

        x_train = self.catalog["normedwaves"][self.train_cat.index.values]
        x_test = self.catalog["normedwaves"][self.test_cat.index.values]
        y_train = self.catalog["labels"][self.train_cat.index.values]
        y_test = self.catalog["labels"][self.test_cat.index.values]
        return x_train, x_test, y_train, y_test