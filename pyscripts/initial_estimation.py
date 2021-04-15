#!/usr/bin/env python
# coding: utf-8

import os

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import odeint
from scipy.optimize import minimize

class FittingInitial:
    """
    Fitting of the curve in the initial days to obtain the estimated values
    for day day march, 16th, 2020, of the pandemic.
    """

    def __init__(self, p, initial_day = '2020-03-16', hmax = 0.1): 

        # parameters pre-determined 
        self.p = p 
        self.hmax = hmax

        # Reading data 
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) 
        filename = os.path.join(ROOT_DIR, "../data/covid_data_organized.csv")
        df = pd.read_csv(filename, index_col = 0)
        self.T = df['confirmed'].loc['2020-03-02':initial_day].to_numpy()
        self.tf = len(self.T)

    def derivative(self, x, t, theta): 
        """
        System of derivatives simplified.
        """
        
        alpha, beta = theta[0], theta[1]
        tau, sigma, gamma1, gamma2 = self.p
        dx = np.zeros(shape = (len(x),))

        dx[0] = -tau*x[0] + beta*(x[1] + x[2]) # E
        dx[1] = tau*x[0] - sigma*x[1]          # I
        dx[2] = sigma*alpha*x[1] - gamma1*x[2] # A
        dx[3] = sigma*(1 - alpha)*x[1]         # T

        dx[4] = dx[3] - gamma2*x[4]            # Q
        dx[5] = gamma1*x[2] + gamma2*x[4]      # R 

        return dx 

    def integrate(self, theta):
        """
        Integrate the system given a tuple of parameters.
        """ 
        y0 = [theta[2], theta[3], theta[4], self.T[0], self.T[0], 0]
        y = odeint(func = self.derivative, 
                   y0 = y0, 
                   t = range(self.tf), 
                   args = (theta,), 
                   hmax = self.hmax)
        return y

    def fit(self, x0, bounds):
        """
        Fits the model to the data and recover the estimated parameters. 
        """
        def objective(theta): 
            series = (self.T - self.integrate(theta)[:,3])**2
            weights = np.linspace(0, 1, len(series))
            obj = 100*sum(series*weights)
            return obj

        res = minimize(fun = objective, 
                       x0 = x0,
                       method='trust-constr',
                       bounds=bounds)

        return res.x

    def get_initial_values(self, x0, bounds): 
        """
        Return E0, I0 and A0 for the system. 
        """
        theta = self.fit(x0, bounds)
        self.y = self.integrate(theta)
        return self.y[-1] # E0, I0, A0, T0, Q0, R0
        
# if __name__ == '__main__': 

#     p = [0.3125, 0.5, 1/9.5, 1/18]
#     bounds = [(0,1), (0,1), (0,0.0001), (0,0.0001), (0,0.0001)] # bound the parameters
#     x0 = [0.8, 0.3, 1e-6, 1e-6, 1e-6]  # initial guess
#     model = FittingInitial(p)
#     theta = model.fit(x0, bounds)
#     print(theta)
#     y = model.diary(model.integrate(theta)[:,3])

#     plt.plot(y)
#     plt.plot(model.T)

#     plt.show()

