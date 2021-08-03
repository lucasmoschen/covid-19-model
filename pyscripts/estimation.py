#!/usr/bin/env python
# coding: utf-8

import os
import time

import pandas as pd 
import numpy as np

from scipy.integrate import odeint
from scipy.optimize import minimize
from scipy.interpolate import BSpline 
from scipy.optimize import approx_fprime

from initial_estimation import FittingInitial

class Fitting:
    """
    Fitting of the model's curve to Rio data to obtain the estimated values for
    the parameters in the initial period of the pandemic.
    p: (tau, sigma, rho, delta, gamma1, gamma2).
    time_varying: definitions about beta and mu bspline (knots, number of parameters and order)
    hmax: max value Runge Kutta integration method. 
    """

    def __init__(self, p, time_varying,    
                initial_day = '2020-03-16', final_day = '2020-07-15', hmax = 0.15, 
                init_cond = {'x0': [0.8, 0.3, 0.00001, 0.00001, 0.00001], 
                             'bounds': [(0,1), (0,1), (0,0.0001), (0,0.0001), (0,0.0001)]} 
                ):

        # parameters pre-determined 
        self.p = np.array(p, dtype = np.float64) 
        self.hmax = hmax
        self.init_cond = init_cond
        self.initial_day = initial_day
        self.final_day = final_day

        # Reading data 
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) 
        filename = os.path.join(ROOT_DIR, "../data/covid_data_organized.csv")
        df = pd.read_csv(filename, index_col = 0)
        self.T = df['confirmed'].loc[initial_day:final_day].to_numpy()
        self.D = df['deaths'].loc[initial_day:final_day].to_numpy()
        self.tf = len(self.T)
        
        # time-varying hyperparameters
        self.sbeta = time_varying['beta']['coefficients']
        self.order_beta = time_varying['beta']['bspline_order']
        self.smu = time_varying['mu']['coefficients']
        self.order_mu = time_varying['mu']['bspline_order']
        self.knots_beta = np.linspace(0, self.tf, self.sbeta + self.order_beta + 1)
        self.knots_mu = np.linspace(0, self.tf, self.smu + self.order_mu + 1)
        # define the time-varying parameters
        self.beta = BSpline(self.knots_beta, np.zeros(self.sbeta), self.order_beta)
        self.mu = BSpline(self.knots_mu, np.zeros(self.smu), self.order_mu)

        # Calculate initial conditions
        print('Model SEIAQR for Covid-19')
        print('-------------------------')
        print('Estimating initial Conditions...')
        self.initial_conditions()
        print('Initiation done!')

    def derivative(self, x, t, alpha, beta_, mu_, tau, sigma, rho, delta, gamma1, gamma2): 
        """
        System of derivatives simplified.
        """

        beta = max(beta_(t), 0)
        mu = max(mu_(t), 0)

        dx = np.zeros(shape = (len(x),))
        dx[4] = -beta*x[4]*(x[1] + x[2])
        dx[0] = -dx[4] - (rho*delta + tau)*x[0]
        dx[1] = tau*x[0] - (sigma + rho)*x[1]
        dx[2] = sigma*alpha*x[1] - (gamma1 + rho)*x[2]
        dx[5] = gamma1*x[2] + gamma2*x[3]
        dx[6] = mu*x[3]
        dx[7] = sigma*(1 - alpha)*x[1] + rho*(delta*x[0] + x[1] + x[2])
        dx[3] = dx[7] - gamma2*x[3] - dx[6]

        return dx 

    def integrate(self, theta, p, time = []):
        """
        Integrate the system given a tuple of parameters.
        p is the parameters estimated by the literature. time is a list always
        with 0 that indicates where to integrate. 
        """ 
        if len(time) == 0: 
            time = range(self.tf)
        self.beta = self.beta.construct_fast(self.knots_beta, theta[1:1+self.sbeta], self.order_beta)
        self.mu = self.mu.construct_fast(self.knots_mu, theta[-self.smu:], self.order_mu)
        states = odeint(func = self.derivative, 
                        y0 = self.y0, 
                        t = time, 
                        args = (theta[0], self.beta, self.mu, *p), 
                        hmax = self.hmax)
        return states

    def rt_calculation(self, theta):
        """
        Calculate the reproduction number based on the model.
        """
        S = self.states[:, 4]
        repro_number = np.zeros(shape = (2, self.tf))

        beta_ = self.beta.construct_fast(self.knots_beta, theta[1:1+self.sbeta], self.order_beta)
        #mu_ = self.mu.construct_fast(self.knots_mu, theta[-self.smu:], self.order_mu)
        alpha = theta[0]
        tau, sigma, rho, delta, gamma1, _ = self.p

        for t in range(self.tf): 
            beta = max(beta_(t), 0)
            #mu = max(mu_(t),0)
            varphi = np.array([beta*tau, beta*tau*S[t]]) # difference between R0 and Rt
            varphi /= ((rho*delta + tau)*(sigma + rho))
            r0_rt = 1/2*(varphi + np.sqrt(varphi**2 + varphi*(4*sigma*alpha)/(rho + gamma1)))
            repro_number[:,t] = r0_rt

        return repro_number

    def initial_conditions(self): 
        """
        Estimate Initial Conditions
        """
        parameters = self.p[[0,1,4,5]]
        model = FittingInitial(parameters, self.initial_day, self.hmax)
        E0, I0, A0, _, Q0, R0 = model.get_initial_values(self.init_cond['x0'], self.init_cond['bounds'])
        self.initial_phase = model.y
        T0 = self.T[0]
        D0 = self.D[0]
        S0 = 1 - E0 - I0 - A0 - Q0 - R0
        self.y0 = [E0, I0, A0, Q0-D0, S0, R0, D0, T0]

    def objective(self, theta, psi, curve1, curve2):
        # theta = (alpha, beta_1, ..., beta_s, mu_1, ..., mu_r)
        integrate = self.integrate(theta, self.p)
        obj1 = (curve1 - integrate[:,7])@self.weights@(curve1 - integrate[:,7])
        obj2 = (curve2 - integrate[:,6])@self.weights@(curve2 - integrate[:,6])

        obj = 100*(obj1 + psi*obj2)
        
        return obj

    def fit(self, psi, x0, bounds, algorithm = 'L-BFGS-B'):
        """
        Fits the model to the data and recover the estimated parameters. 
        """
        self.weights = np.array([[min(i,j)+15 for i in range(self.tf)] for j in range(self.tf)])
        self.weights = np.linalg.inv(self.weights)

        print('Starting estimation!')
        t0 = time.time()
        res = minimize(fun = self.objective, 
                       x0 = x0,
                       method=algorithm,
                       bounds=bounds, 
                       args=(psi,self.T,self.D))
        self.counter = time.time() - t0
        print('Estimation finished. It took {} seconds'.format(self.counter))

        self.states = self.integrate(res.x, self.p)

        # Rt calculation
        self.repro_number = self.rt_calculation(res.x)
        
        # Store important values 
        self.obj = res.fun
        self.res = res
        self.theta = res.x
        self.psi = psi
        self.x0 = x0
        self.bounds = bounds
        self.algorithm = algorithm
        
        n = self.tf
        K = len(self.theta)

        # Estimate variances
        self.sigma2_1 = (self.T - self.states[:,7])@self.weights@(self.T - self.states[:,7])/(n-K)
        self.sigma2_2 = (self.D - self.states[:,6])@self.weights@(self.D - self.states[:,6])/(n-K)  
        
        # Information Criterion
        common = n*np.log(self.obj/n)
        self.aic = common + 2*K
        self.bic = common + np.log(n)*K
        self.aicc = common + 2*K*n/(n - K - 1)

        return res.x
    
    def fit_fast(self, curveT, curveD, x0):
        """
        Simpler fit function.
        """
        res = minimize(fun = self.objective, 
                       x0 = x0,
                       method=self.algorithm,
                       bounds=self.bounds, 
                       args=(self.psi,curveT, curveD))
        return res

    def check_residuals(self):
        """
        Simple residual analysis for the fitting. It must be called after the
        function fit. 
        """
        diary_curves = self.integrate(self.theta, self.p)

        T = diary_curves[:,7]
        D = diary_curves[:,6]
        
        errorT = np.diff(self.T - T)
        errorD = np.diff(self.D - D)
        
        return errorT, errorD

    def correlation_matrix(self):
        """
        Calculate the correlation matrix with an estimated parameter. It must be called after the
        function fit. 
        """
        def f(parameters, time, curve): 
            theta = parameters[0:len(self.theta)]
            #p = parameters[len(self.theta):]
            return self.integrate(theta, self.p, [0,time])[1,curve]

        K = len(self.theta) #+ len(self.p)
        J1 = np.zeros((self.tf,K))
        J2 = np.zeros((self.tf,K))
        parameters = self.theta #np.concatenate([self.theta, self.p])
        for i in range(self.tf): 
            J1[i,:] = approx_fprime(parameters, f, np.ones_like(parameters)*1e-5, i, 7)
            J2[i,:] = approx_fprime(parameters, f, np.ones_like(parameters)*1e-5, i, 6)
            
        # Fisher Information matrix
        FIM = J1.transpose()@self.weights@J1/self.sigma2_1 + J2.transpose()@self.weights@J2/self.sigma2_2
        # Covariance matrix
        C = np.linalg.inv(FIM)
        # Correlation matrix
        R = [[C[i,j]/np.sqrt(C[i,i]*C[j,j]) for i in range(K)] for j in range(K)]

        return np.array(R)
                    
    def _get_exp(self, pathname): 

        with open(pathname, 'r') as f: 
            line = f.readline()
            while line != '':
                lineold = line
                line = f.readline()
        exp = lineold[:lineold.find(';')]
        exp = 1 if exp == 'exp' else int(exp) + 1 

        return exp

    def save_experiment(self, objective_function): 
        """
        Save information about the experiment. 
        objective_function: name given to compare, like quadratic and divided.
        """
        pathname = '../experiments/' + objective_function + '.csv'
        if not os.path.exists(pathname): 
            with open(pathname, 'w') as f: 
                f.write('exp;tau;sigma;rho;delta;gamma1;gamma2;sbeta;order_beta;smu;order_mu;')
                f.write('initial_day;final_day;hmax;psi;x0;bounds;algorithm;')
                f.write('E0;I0;A0;Q0;S0;R0;D0;T0;alpha;beta;mu;obj;time')
                f.write('\n')
        else: 
            with open(pathname, 'a') as f: 
                exp = self._get_exp(pathname)
                tau, sigma, rho, delta, gamma1, gamma2 = self.p 

                info = [exp,tau,sigma,rho,delta,gamma1,gamma2,self.sbeta,self.order_beta,self.smu,self.order_mu]
                info2 = [self.initial_day,self.final_day,self.hmax,self.psi]

                f.write(';'.join(map(str,info)))
                f.write(';')
                f.write(';'.join(map(str,info2)))
                f.write(';')
                f.write(str(self.x0))
                f.write(';')
                f.write(str(self.bounds))
                f.write(';')
                f.write(self.algorithm + ';')
                f.write(';'.join(map(str, self.y0)))
                f.write(';' + str(self.theta[0]) + ';')
                f.write(str(self.theta[1:1+self.sbeta]))
                f.write(';')
                f.write(str(self.theta[-self.smu:]))
                f.write(';')
                f.write(str(self.obj) + ';')
                f.write(str(self.counter))
                f.write('\n')
        
        
if __name__ == '__main__': 

    tau    = 1/3.69
    omega  = 1/5.74
    sigma  = 1/(1/omega - 1/tau)
    rho    = 1e-5
    delta  = 0.01
    gamma1 = 1/7.5
    gamma2 = 1/13.4
    psi = 119.56329270977567

    p = [tau, sigma, rho, delta, gamma1, gamma2]

    # sbeta is number of coefficients, bspline order speaks for itself
    time_varying = {'beta': {'coefficients':  4, 'bspline_order': 3}, 
                'mu'  : {'coefficients': 4, 'bspline_order': 3}}

    # initial and final day in the model
    initial_day = '2020-03-16'
    final_day = '2020-07-31'

    # initial conditions guesses and bounds
    init_cond = {'x0': [0.8, 0.3, 5e-7, 5e-7, 5e-7], 
                'bounds': [(0.5,1), (0,1), (1e-7, 5e-5), (1e-7, 5e-5), (1e-7, 5e-5)]}
    # hmax
    hmax = 0.2

    # bound the parameters
    bounds = [(0.7, 0.95), 
            (0.05, 0.2), (0.05, 0.2), (0.05, 0.2), (0.05, 0.2), 
            (0.005, 0.02), (0.005,0.02), (0.005, 0.02), (0.005, 0.02)] 

    # initial guess
    x0 = [0.9,                        # alpha
        0.11, 0.08, 0.09, 0.12,     # beta
        0.015, 0.015, 0.011, 0.009] # mu

    # defines the model
    model = Fitting(p, time_varying, initial_day, final_day, hmax, init_cond)

    # estimating theta
    theta = model.fit(psi, x0, bounds)