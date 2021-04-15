#!/usr/bin/env python
# coding: utf-8

import os
import time

import numpy as np

from estimation import Fitting

class BootstrapMethod(Fitting): 
    """
    Parametric Bootstrap generates simulated curves and estimate the
    parameters for each curve. Report the confidence intervals produced. 
    - random_state: integer to guarantee reproducibility. 
    - parameters for the Fitting class. 
    - filename: file to save bootstrap information.
    """
    def __init__(self, random_state, p, time_varying, initial_day, final_day, hmax, init_cond, 
                       psi, x0, bounds, filename):     
        
        super().__init__(p, time_varying, initial_day, final_day, hmax, init_cond)
        self.ro = np.random.RandomState(seed = random_state)
        _ = self.fit(psi, x0, bounds)
        if not os.path.exists(filename):
            self.file = open(filename, 'w')
            self.file.write('seed;obj;alpha;beta;mu\n')
        else: 
            self.file = open(filename, 'a')

    def generate_curve(self, curve, sigma2_hat, ro): 
        """
        Simulates a curve given the error distribution and an estimation for
        sigma2_hat. `ro` is a RandomState.  
        """
        new = np.zeros_like(curve)
        new[0] = self.T[0]
        for i in range(self.tf-1):
            new[i+1] = new[i] + ro.normal(loc = curve[i+1]-curve[i], scale = np.sqrt(sigma2_hat))
        return new

    def bootstrap_simulation(self, N):
        """
        Generates the curves and estimates the parameter for each one. Save
        each experiment to avoid losses. 
        """
        for exp in range(N): 
            # Simulated experiment curves
            print("Initiating experiment {}".format(exp))
            seed = self.ro.randint(N**3)
            ro = np.random.RandomState(seed = seed)
            T_new = self.generate_curve(self.T, self.sigma2_1, ro)
            D_new = self.generate_curve(self.D, self.sigma2_2, ro)

            res = self.fit_fast(T_new, D_new)

            self.save_bootstrap_experiment(seed, res.fun, res.x)

        self.file.close()

    def save_bootstrap_experiment(self, seed, obj, parameters):
        """
        Save important information about bootstrap's experiments. 
        """
        seed = str(seed)
        obj = str(obj)
        alpha = str(parameters[0])
        beta = str(parameters[1:1+self.sbeta])
        mu = str(parameters[-self.smu:])
        self.file.write(seed + ';' + obj + ';' + alpha + ';' + beta + ';' + mu + '\n') 


if __name__ == '__main__': 

    tau    = 1/3.69
    omega  = 1/5.74
    sigma  = 1/(1/omega - 1/tau)
    rho    = 1e-5
    delta  = 0.01
    gamma1 = 1/7.5
    gamma2 = 1/13.4
    psi = 119.571       

    p = [tau, sigma, rho, delta, gamma1, gamma2]

    time_varying = {'beta': {'coefficients':  4, 'bspline_order': 2}, 
                'mu'  : {'coefficients': 4, 'bspline_order': 1}}

    initial_day = '2020-03-16'
    final_day = '2020-07-31'

    init_cond = {'x0': [0.8, 0.3, 5e-7, 5e-7, 5e-7], 
                'bounds': [(0.5,1), (0,1), (1e-7, 5e-5), (1e-7, 5e-5), (1e-7, 5e-5)]}
    hmax = 0.2

    bounds = [(0.7, 0.95), 
            (0.05, 0.2), (0.05, 0.2), (0.05, 0.2), (0.05, 0.2), 
            (0.005, 0.02), (0.005,0.02), (0.005, 0.02), (0.005, 0.02)] 

    x0 = [0.9,                      
         0.11, 0.08, 0.09, 0.12,     
         0.015, 0.015, 0.011, 0.009] 

    random_state = 73812990
    filename = "../experiments/bootstrap1.csv"

    samples = BootstrapMethod(random_state, p, time_varying, initial_day, final_day, 
                              hmax, init_cond, psi, x0, bounds, filename)

    N = 100
    samples.bootstrap_simulation(N)