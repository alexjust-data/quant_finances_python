# !/usr/bin/ python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import scipy
import importlib
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, chi2
from scipy.optimize import minimize
from numpy import linalg as LA



def load_time_series(ric, file_extension="csv"):
    """Get markets data
    Arguments:
        - ric : cod of market
    Return:
        - x : array of real returns for day
        - x_str : real returns daily ric
        - t : dataframe
    """

    path = "data/" + ric + "." + file_extension
    if file_extension == "csv":
        table_raw = pd.read_csv(path) 
    else:
        table_raw = pd.read_excel(path) 

    # creating table returns
    t = pd.DataFrame()
    t['date'] = pd.to_datetime(table_raw['Date'], dayfirst=True)
    t['close'] = table_raw['Close']            # close price market
    t.sort_values(by="date", ascending=True)   # assure order days
    t['close_previous'] = table_raw['Close'].shift(1)
    t['return_close'] = (t['close'] / t['close_previous']) - 1
    t = t.dropna()                             # eliminate nan
    t = t.reset_index(drop=True)               # for having deleted rows

    # input for Jarque-Bera test
    x = t['return_close'].values               # return to arrays
    x_str = "Real returns " + ric

    return x, x_str, t



def jarque_bera_test(x, x_str):
    """Create a normality test [by Jarque_Bera]
    Arguments:
        - x : array of real returns for day
        - x_str : real returns daily ric
    Return:
        - plot_str : str with risk mesuares
    """
    x_mean = np.mean(x)
    x_stdev = np.std(x)
    x_skew = skew(x)
    x_kurt = kurtosis(x)
    x_sharpe = x_mean / x_stdev * np.sqrt(252) # anualizado
    x_var_95 = np.percentile(x, 5)
    x_cvar_95 = np.mean(x[x <= x_var_95])
    jb = len(x)/6 * (x_skew**2 + 1/4*(x_kurt**2))
    p_value = 1 - chi2.cdf(jb, df=2) # buscamos dos grados de libertad
    is_normal = (p_value > 0.05) # equivalenty jb < 6

    # print risk metrics
    plot_str = "mean: " + str(np.round(x_mean, 4))\
        + " | std : " + str(np.round(x_stdev, 4))\
        + " | skewness : " + str(np.round(x_skew, 4))\
        + " | kurtosis : " + str(np.round(x_kurt, 4))\
        + " | x_sharpe ratio :" + str(np.round(x_sharpe, 4)) + "\n"\
        + " VaR 95% : " + str(np.round(x_var_95, 4))\
        + " | CVaR 95% : " + str(np.round(x_cvar_95, 4))\
        + " | Jarque-Bera : " + str(np.round(jb, 4))\
        + " | p_value : " + str(np.round(p_value, 4))\
        + " | is normal :" + str(is_normal)

    return plot_str



def plot_time_series_price(ric, t):
    """Plot series time from data markets
    Arguments:
       -ric : cod  
       -t :  dataframe
    """
    plt.figure()
    plt.plot(t["date"], t["close"])
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.title("tiem series real price " + ric)




def plot_histogram(x, x_str, plot_str, bins=100):
    """Plot histogram from data markets
    Arguments:
       -x : cod  
       -x_str : dataframe
       -str1 : print risk metrics
       -str2 : print risk metrics
    """
    plt.figure()
    plt.hist(x, bins)
    plt.title("Histrogram " + x_str)
    plt.xlabel(plot_str)
    plt.show()