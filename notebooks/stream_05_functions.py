# !/usr/bin/ python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import scipy
import importlib
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import skew, kurtosis, chi2, linregress

# import our own files an reload
import stream_classes_refactoring2
import stream_functions
importlib.reload(stream_functions)
importlib.reload(stream_classes_refactoring2)



def camp(ric, benchmark, file_extension, nb_decimals):
    """Genera plot de regresión lineal con medidas de riesgo
    Arguments:
        - ric : valor de mercado
        - benchmark : mercado
        - file_extension : per defeco cvs
        - nb_decimals : decimales de ajuste, por defecto 4
    Returns:
        - Regresión lineal y medidas de riesgo.
    """

    # ric = r_a and benchmark = r_m
    x1, str1, t1 = stream_functions.load_time_series(ric)
    x2, str2, t2 = stream_functions.load_time_series(benchmark)

    # sinchronize timestamps
    timestamps1 = list(t1['date'].values) # .values lo convierte en array
    timestamps2 = list(t2['date'].values)
    timestamps = list(set(timestamps1) & set(timestamps2))

    # synchronised time series for x1 (ric)
    t1_sync = t1[t1['date'].isin(timestamps)]
    t1_sync.sort_values(by='date', ascending=True)
    t1_sync = t1_sync.reset_index(drop=True)

    # synchronised time series for x2 (benchmark)
    t2_sync = t2[t2['date'].isin(timestamps)]
    t2_sync.sort_values(by='date', ascending=True)
    t2_sync = t2_sync.reset_index(drop=True)

    # table of returns for ric and benchmark
    t = pd.DataFrame()
    t['date'] = t1_sync.date
    t['price_1'] = t1_sync.close
    t['price_2'] = t2_sync.close
    t['return_1'] = t1_sync.return_close
    t['return_2'] = t2_sync.return_close

    # compute vectors of return
    y = t['return_1'].values
    x = t['return_2'].values

    # linal regression
    slope, intercep, r_values, p_values, std_err = linregress(x, y)

    slope = np.round(slope, nb_decimals)
    intercept = np.round(intercep, nb_decimals)
    p_values = np.round(p_values, nb_decimals)
    r_values = np.round(r_values, nb_decimals)
    r_squared = np.round(r_values**2, nb_decimals)

    # Si el `p_value < 0.05` rechazamos la hipótesis nula.
    null_hypothesis = p_values > 0.05

    # predictor 
    predictor_linreg = slope*x + intercept

    # scatterplot of returns
    str_title = "Scarterplot of returns " + "\n"\
        + "Linear regression of / ric : " + ric\
        + " / benchmark : " + benchmark + "\n"\
        + "alpha (intercept) : " + str(intercept)\
        + " / beta (slope) : " + str(slope) + "\n"\
        + "p-value : " + str(p_values)\
        + " / null hypothesis : " + str(null_hypothesis) + "\n"\
        + "r-value : " + str(r_values)\
        + " / r-squared : " + str(r_squared)

    plt.figure()
    plt.title(str_title)
    plt.scatter(x,y)
    plt.plot(x, predictor_linreg, color="green")
    plt.ylabel(ric)
    plt.xlabel(benchmark)
    plt.grid()
    plt.show()