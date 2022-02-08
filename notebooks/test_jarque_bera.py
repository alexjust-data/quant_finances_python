# !/usr/bin/ python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import scipy
import importlib
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import skew, kurtosis, chi2


def normality_test(x_size, type_random_variable, degrees_freedom=5):
    """Create a normality test by example with Jarque_Bera
    Arg:
        1. x_size : size of distribution.
        2. type_random_variable : generate random distributions.
        3. degrees_freedom : only for t-student
    Return:
        Plot and risk metrics
    """
    if type_random_variable == "normal":
        x = np.random.standard_normal(x_size)
        x_str = type_random_variable
    elif type_random_variable == "exponential":
        x = np.random.standard_exponential(x_size)
        x_str = type_random_variable
    elif type_random_variable == "student":
        x = np.random.standard_t(size=x_size, df=degrees_freedom)
        x_str = type_random_variable + "(df=" + str(degrees_freedom) + ")"
    elif type_random_variable == "chi-squared":
        x = np.random.chisquare(size=x_size, df=degrees_freedom)
        x_str = type_random_variable + "(df=" + str(degrees_freedom) + ")"

    # compute risk metrics
    x_mean = np.mean(x)
    x_stdev = np.std(x)
    x_skew = skew(x)
    x_kurt = kurtosis(x)
    x_var_95 = np.percentile(x, 5)
    x_cvar_95 = np.mean(x[x <= x_var_95])
    jb = x_size/6 * (x_skew**2 + 1/4*(x_kurt**2))
    p_value = 1 - chi2.cdf(jb, df=2) # buscamos dos grados de libertad
    is_normal = (p_value > 0.05) # equivalenty jb < 6

    # plot histogram
    plt.figure()
    plt.hist(x, bins=100)
    plt.title(type_random_variable)
    plt.show()

    # print risk metrics
    print("Distibución: " + x_str.upper())
    print("mean: " + str(x_mean))
    print("std : " + str(x_stdev))
    print("skewness : " + str(x_skew))
    print("kurtosis : " + str(x_kurt))
    print("VaR 95% : " + str(x_var_95))
    print("CVaR 95% : " + str(x_cvar_95))
    print("Jarque-Bera : " + str(jb))
    print("p_value : " + str(p_value))
    print("is normal :" + str(is_normal))


if __name__ == "__main__":

    normality_test(10**6, "normal")