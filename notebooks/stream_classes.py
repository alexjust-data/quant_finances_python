# !/usr/bin/ python
# -*- coding: utf-8 -*-

import scipy
import importlib
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import minimize
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
    
    def __str__(self):
        str_self = self.str_name + " | size " + str(self.size) + \
                   "\n" + self.plot_str() + "\n"
        return str_self

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
    
    def plot_timeseries(self):
        """Plot series time from data markets
        Arguments:
           -ric : cod  
           -t :  dataframe
        """
        plt.figure()
        plt.plot(self.t["date"], self.t["close"])
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.title("tiem series real price " + self.ric)

    def plot_histogram(self):
        """Plot histogram from data markets
        Arguments:
           -x : cod  
           -x_str : dataframe
           -str1 : print risk metrics
           -str2 : print risk metrics
        """
        self.bins=100
        plt.figure()
        plt.hist(self.returns, self.bins)
        plt.title("Histrogram " + self.str_name)
        plt.xlabel(self.plot_str())
        plt.show()


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
    
    def plot_normalised(self):
        # plot 2 timeseries normalised at 100
        timestamps = self.t['date']
        price_ric = self.t['price_1']
        price_benchmark = self.t['price_2']

        plt.figure(figsize=(12,5))
        plt.title('Tine series of prices | normalised at 100')
        plt.ylabel('Time')
        plt.xlabel('Normalised prices')

        price_ric = 100 * price_ric / price_ric[0]
        price_benchmark = 100 * price_benchmark / price_benchmark[0]

        plt.plot(timestamps, price_ric, color='blue', label=self.ric)
        plt.plot(timestamps, price_benchmark, color='red', label=self.benchmark)
        plt.legend(loc=0)
        plt.grid()
        plt.show
        
    def plot_dual_axes(self):
        # plot 2 timeseries with 2 vertical axes
        plt.figure(figsize=(12,5))
        plt.title('Time series of price')
        plt.xlabel('Time')
        plt.ylabel('Prices')
        # creo un eje horizontal para las dos
        ax = plt.gca()
        # t es un df y uso pandas plot
        ax1 = self.t.plot(kind='line', x='date',
                                       y='price_1',
                                       ax=ax,
                                       color='blue',
                                       grid=True,
                                       label=self.ric)
        ax2 = self.t.plot(kind='line', x='date',
                                       y='price_2',
                                       ax=ax,color='red',
                                       grid=True,
                                       label=self.benchmark,
                                       secondary_y=True)
        ax1.legend(loc=2)
        ax2.legend(loc=1)
        plt.show()


class hedge_maneger():
    
    def __init__(self, ric, benchmark, hedge_rics, delta):
        self.ric = ric
        self.benchmark = benchmark
        self.hedge_rics = hedge_rics
        self.delta = delta
        self.dataframe= pd.DataFrame()            # quiero un df con el resumen de la cobertura
        self.hedge_delta = None                   # para cálculo del algortimo
        self.hedge_beta_usd = None                # para cálculo del algortimo
    
    def load_inputs(self, bool_print=False):
        self.beta = stream_functions.compute_beta( # llamo función creada para calcular beta
            self.ric, self.benchmark
        )
        self.beta_usd = self.beta * self.delta
        betas = [                           # Calcula betas de cada valor de las coberturas
            stream_functions.compute_beta(i, self.benchmark) for i in self.hedge_rics
        ]
        self.betas = np.array(betas).reshape([len(self.hedge_rics), 1])
        self.dataframe['ric'] = self.hedge_rics  # cargando columnas de mi df de salida
        self.dataframe['beta'] = betas           # cargando columnas de mi df de salida
        if bool_print:
            print('---------------')
            print('Imput portfolio')
            print('---------------')
            print('Delta mnUSD for ' + self.ric + ' is ' + str(self.delta))
            print('Beta for ' + self.ric + ' vs ' + self.benchmark + ' is ' + str(self.beta))
            print('Beta mnUSD for ' + self.ric + ' vs ' + self.benchmark + ' is ' + str(self.beta_usd))
            print('---------------')
            print('Input hedges:')
            print('-------------')
            # a cada vector le calculo el beta para cada ric de cobertura
            for i in range(self.dataframe.shape[0]):
                print('Beta for hedge[ ' + str(i) + '] = ' + self.dataframe['ric'][i] \
                      + ' vs ' + self.benchmark + ' is ' + str(self.dataframe['beta'][i]))
    
    def compute(self, bool_print=False):
        size = len(self.hedge_rics)      
        if not size == 2:
            print('-------')
            print('Warning: cannot compute exact solution, hedge rics size ' + str(size) + ' =/= 2')
            return
        deltas = np.ones([size, 1])                 # matriz de unos
        targets = - np.array([[self.delta],[self.beta_usd]])
        mtx = np.transpose(                         # traspuesta
                np.column_stack(                    # acomodo como columnas
                    (deltas, self.betas)))          # relleno mis columnas
        self.optimal_hedge = np.linalg.inv(mtx).dot(targets)
        self.dataframe['delta'] = self.optimal_hedge
        self.dataframe['beta_usd'] = self.betas * self.optimal_hedge # dot of matrix
        self.hedge_delta = np.sum(self.dataframe['delta'])
        self.hedge_beta_usd = np.sum(self.dataframe['beta_usd'])
        if bool_print:
            self.print_output('Exact solution from linear algebra')
            
    def compute_numerical(self, epsilon=0.0, bool_print=False):
        x = np.zeros([len(self.betas), 1]) # núm de betas equivalente al num de raíces
        args = (self.delta, # el delta del portafolios
                self.beta_usd, # beta del portfolio en dólares
                self.betas, # vector con betas de los activos de cobertura
                epsilon)
        optimal_result = minimize(fun=stream_functions.cost_function_beta_delta,
                                  x0=x, args=args, method='BFGS')
        self.optimal_hedge = optimal_result.x.reshape([len(self.betas), 1])
        self.dataframe['delta'] = self.optimal_hedge
        self.dataframe['beta_usd'] = self.betas * self.optimal_hedge
        self.hedge_delta = np.sum(self.dataframe['delta'])
        self.hedge_beta_usd = np.sum(self.dataframe['beta_usd'])
        if bool_print:
            self.print_output('Numerical solution with optimize.minimize')
        
    def print_output(self, optimisation_type):
            print('-------------------')
            print('Optimisation result ' + optimisation_type + ':')
            print('-------------------')
            print('Delta: ' + str(self.delta))
            print('Beta USD: ' + str(self.beta_usd))
            print('')
            print('Hedge delta:' + str(self.hedge_delta))
            print('Hedge beta:' + str(self.hedge_beta_usd))
            print('--------------------')
            print('Betas for the hedge:')
            print('--------------------')
            print(self.betas)
            print('--------------')
            print('Optimal hedge:')
            print('--------------')
            print(self.optimal_hedge)