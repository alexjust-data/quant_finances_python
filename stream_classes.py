# !/usr/bin/ python
# -*- coding: utf-8 -*-

import scipy
import importlib
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, chi2, linregress
# load our own file
import stream_functions
importlib.reload(stream_functions)


class jarque_bera_test():

    def __init__(self, ric):
        # create constructor
        self.ric = ric
        self.returns = []
        self.size = 0
        self.str_name = None
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
        self.round_digit = 4

    def load_timeseries(self):
        self.returns, self.str_name, self.t = stream_functions.load_time_series(self.ric)
        self.size = self.t.shape[0]

    def compute(self):
        # compute "risk metrics"
        self.size = self.t.shape[0]
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


class capm_manager():
    # constructor
    def __init__(self, ric, benchmark):
        self.nb_decimals = 4
        self.ric = ric
        self.benchmark = benchmark
        self.x = []
        self.y = []
        self.t = pd.DataFrame()
        self.beta = 0.0
        self.alpha = 0.0
        self.p_values = 0.0
        self.null_hypothesis = False
        self.r_values = 0.0
        self.r_squared = 0.0
        self.predictor_linreg = []
    
    def __str__(self):
        str_self = "Linear regression of | ric : " + self.ric\
            + " | benchmark : " + self.benchmark + "\n"\
            + "alpha (intercept) : " + str(self.alpha)\
            + " | beta (slope) : " + str(self.beta) + "\n"\
            + "p-value : " + str(self.p_values)\
            + " | null hypothesis : " + str(self.null_hypothesis) + "\n"\
            + "r-value : " + str(self.r_values)\
            + " | r-squared : " + str(self.r_squared)
        return str_self
    
    def load_timeseries(self):
        # load timeseries and syncronise them
        self.x, self.y, self.t = stream_functions.synchronise_timeseries(self.ric, self.benchmark)
    
    def compute(self):
        # linal regression of ric with respect to becnhmark
        slope, intercep, r_values, p_values, std_err = linregress(self.x, self.y)
        self.beta = np.round(slope, self.nb_decimals)
        self.alpha = np.round(intercep, self.nb_decimals)
        self.p_values = np.round(p_values, self.nb_decimals)
        self.null_hypothesis = p_values > 0.05
        self.r_values = np.round(r_values, self.nb_decimals)
        self.r_squared = np.round(r_values**2, self.nb_decimals)
        self.predictor_linreg = self.alpha + self.beta*self.x
    
    def scatterplot(self):
        # scatterplot of returns
        str_title = "Scarterplot of returns " + self.__str__()
        plt.figure()
        plt.title(str_title)
        plt.scatter(self.x,self.y)
        plt.plot(self.x, self.predictor_linreg, color="green")
        plt.ylabel(self.ric)
        plt.xlabel(self.benchmark)
        plt.grid()
        plt.show()