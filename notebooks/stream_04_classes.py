# !/usr/bin/ python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import scipy
import importlib
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import skew, kurtosis, chi2

import stream_functions
importlib.reload(stream_functions)  # aveces no importa bien


class jarque_bera_test():

    # create constructor
    # self. nos permitirá llamar a cada uno de ellos
    def __init__(self, x):
        self.size = len(x) #. size of returns
        self.x_mean = np.mean(x)
        self.x_stdev = np.std(x)
        self.x_skew = skew(x)
        self.x_kurt = kurtosis(x)
        self.x_sharpe = self.x_mean / self.x_stdev * np.sqrt(252)  # anualizado
        self.x_var_95 = np.percentile(x, 5)
        self.x_cvar_95 = np.mean(x[x <= self.x_var_95])
        self.jarque_bera = self.size/6 * (self.x_skew**2 + 1/4*(self.x_kurt**2))
        # buscamos dos grados de libertad
        self.p_value = 1 - chi2.cdf(self.jarque_bera, df=2)
        self.is_normal = (self.p_value > 0.05)  # equivalenty jb < 6

        # print risk metrics
        plot_str = "mean: " + str(np.round(self.x_mean, 4))\
            + " | std : " + str(np.round(self.x_stdev, 4))\
            + " | skewness : " + str(np.round(self.x_skew, 4))\
            + " | kurtosis : " + str(np.round(self.x_kurt, 4))\
            + " | x_sharpe ratio :" + str(np.round(self.x_sharpe, 4)) + "\n"\
            + " VaR 95% : " + str(np.round(self.x_var_95, 4))\
            + " | CVaR 95% : " + str(np.round(self.x_cvar_95, 4))\
            + " | Jarque-Bera : " + str(np.round(self.jarque_bera, 4))\
            + " | p_value : " + str(np.round(self.p_value, 4))\
            + " | is normal :" + str(self.is_normal)