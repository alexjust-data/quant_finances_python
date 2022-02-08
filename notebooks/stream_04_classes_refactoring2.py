# !/usr/bin/ python
# -*- coding: utf-8 -*-

import numpy as np
import importlib
from scipy.stats import skew, kurtosis, chi2


class jarque_bera_test():

    def __init__(self, x, x_str):
        # create constructor
        self.returns = x
        self.str_name = x_str
        self.size = len(x)  # size of returns
        self.round_digit = 5  # por si quiero modificarlo
        self.mean = 0.0
        self.stdev = 0.0
        self.skew = 0.0
        self.kurt = 0.0
        self.sharpe = 0.0
        self.median = 0.0
        self.var_95 = 0.0
        self.cvar_95 = 0.0
        self.jarque_bera = 0.0
        self.p_value = 0.0
        self.is_normal = 0.0

    def compute(self):
        # compute "risk metrics"
        self.mean = np.mean(self.returns)
        self.stdev = np.std(self.returns)
        self.skew = skew(self.returns)
        self.kurt = kurtosis(self.returns)
        self.sharpe = self.mean / self.stdev * np.sqrt(252)
        self.median = np.median(self.returns)
        self.var_95 = np.percentile(self.returns, 5)
        self.cvar_95 = np.mean(self.returns[self.returns <= self.var_95])
        self.jarque_bera = self.size/6 * (
            self.skew**2 + 1/4*(self.kurt**2)
        )
        self.p_value = 1 - chi2.cdf(self.jarque_bera, df=2)
        self.is_normal = (self.p_value > 0.05)  # equivalenty jb < 6

    def __str__(self):
        str_self = self.str_name + " | size " + str(self.size) + \
                   "\n" + self.plot_str() + "\n"
        return str_self

    def plot_str(self):
        plot_str = "mean: " + str(np.round(self.mean, self.round_digit))\
            + " | std : " + str(np.round(self.stdev, self.round_digit))\
            + " | skewness : " + str(np.round(self.skew, self.round_digit))\
            + " | kurtosis : " + str(np.round(self.kurt, self.round_digit))\
            + " | median : " + str(np.round(self.median, self.round_digit))\
            + " | x_sharpe ratio :" + str(np.round(self.sharpe, 4)) + "\n"\
            + " VaR 95% : " + str(np.round(self.var_95, 4))\
            + " | CVaR 95% : " + str(np.round(self.cvar_95, self.round_digit))\
            + " | Jarque-Bera : " + str(np.round(self.jarque_bera, self.round_digit))\
            + " | p_value : " + str(np.round(self.p_value, self.round_digit))\
            + " | is normal :" + str(self.is_normal)
        return plot_str