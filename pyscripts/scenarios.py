#!/usr/bin/env python
# coding: utf-8

from execute_model import main
import yaml 
import numpy as np 
from tabulate import tabulate

class Parameters_Functions:

    def __init__(self, values, days_change):
        self.values_beta = values[0]
        self.values_r = values[1]
        self.values_rho = values[2] 

        self.day_change_beta = days_change[0]
        self.day_change_r = days_change[1]
        self.day_change_rho = days_change[2]
        
    def pbeta(self,t):
        if len(self.values_beta) == 1:
            return self.values_beta[0]
        else: 
            if t <= self.day_change_beta: return self.values_beta[0]
            else: return self.values_beta[1]

    def pr(self,t):
        if len(self.values_r) == 1:
            return self.values_r[0]
        else: 
            if t < self.day_change_r: return self.values_r[0]
            else: return self.values_r[1]

    def prho(self,t):
        if len(self.values_rho) == 1:
            return self.values_rho[0]
        else: 
            if t <= self.day_change_rho: return self.values_rho[0]
            else: return self.values_rho[1]

def change_values(parameters: dict, tau, sigma, alpha, delta, epsilon, gamma1, gamma2, mu, p):
    """Update parameters in the dictionary"""

    parameters['tau'] = tau
    parameters['sigma'] = sigma
    parameters['alpha'] = alpha
    parameters['delta'] = delta 
    parameters['epsilon'] = epsilon 
    parameters['gamma1'] = gamma1
    parameters['gamma2'] = gamma2
    parameters['mu'] = mu 
    parameters['p'] = p 

    return parameters

def writing(parameters): 
    with open('../data/parameters.yaml', 'w') as f: 
        yaml.dump(parameters, f) 

def calculates_statistics(file_name):

    with open('../data/variables/'+file_name, 'r') as f: 
        important_values = {}
        line = f.readline()
        while line != '': 
            words = line.split(',') 
            if words[0] in 'QR0TD': 
                value = np.array([float(i) for i in words[1:]])
                if words[0] == 'Q': 
                    important_values['peak_day'] = np.argmax(value)
                    important_values['peak_max'] = np.max(value)
                    tmp = np.where(value <= 1e-9)[0]
                    if len(tmp[tmp > np.argmax(value)]) == 0: 
                        important_values['ending'] = len(value)
                    else: 
                        important_values['ending'] = tmp[tmp > np.argmax(value)][0]
                elif words[0] == 'R':
                    important_values['R'] = value[-1]
                elif words[0] == 'D':
                    important_values['D'] = value[-1]
                elif words[0] == 'T':
                    important_values['T'] = value[-1]
                else: 
                    important_values['R0'] = [value[0], value[-1]]
            line = f.readline()

    with open('../data/table_values.txt', 'a') as f: 
        f.write(file_name)
        f.write('\n')
        f.write(tabulate(important_values.items()))
        f.write('\n\n')
         
    return important_values

if __name__ == '__main__': 

    with open('../data/table_values.txt', 'w') as f:
        pass

    with open('../data/parameters.yaml', 'r') as f: 
        parameters = yaml.load(f, Loader=yaml.FullLoader)
    
    parameters['initial']['E_0'] = 0.000001
    parameters['initial']['S_0'] = 0.999999
    parameters['initial']['I_0'] = 0
    parameters['initial']['A_0'] = 0
    parameters['initial']['Q_0'] = 0
    parameters['initial']['R_0'] = 0
    parameters['initial']['D_0'] = 0
    parameters['initial']['T_0'] = 0
        
    # Scenario A

    # A1
    parameters['Tf'] = 500
    parameters['image_name'] = "scenario_A1_new.svg"
    # Here I insert values of beta, r and rho and the days of change. If it
    # not change, insert None. 
    functions = Parameters_Functions([[0.7676], [1], [0]], [None, None, None])
    parameters['parameters'] = change_values(parameters['parameters'], tau=1/3.2, sigma=1/2, alpha=0.4, delta=1/2, 
                                              epsilon=0, gamma1=1/8, gamma2=1/16, mu=0.058/14, p=[0])
    parameters['change_p'] = 500 
    writing(parameters)
    main(functions, file_name='scenario_A1_new')
    print(calculates_statistics('scenario_A1_new.txt'))

    # A2
    parameters['image_name'] = "scenario_A2_new.svg"
    functions = Parameters_Functions([[0.7676], [1, 0.2], [0]], [None, 31, None])
    parameters['parameters'] = change_values(parameters['parameters'], tau=1/3.2, sigma=1/2, alpha=0.4, delta=1/2, 
                                              epsilon=0, gamma1=1/8, gamma2=1/16, mu=0.058/14, p=[0,0.6])
    parameters['change_p'] = 31
    writing(parameters)
    main(functions, file_name='scenario_A2_new')
    print(calculates_statistics('scenario_A2_new.txt'))

    # A3
    parameters['image_name'] = "scenario_A3_new.svg"
    functions = Parameters_Functions([[0.7676], [1, 0.2], [0]], [None, 31, None])
    parameters['parameters'] = change_values(parameters['parameters'], tau=1/3.2, sigma=1/2, alpha=0.4, delta=1/2, 
                                              epsilon=0, gamma1=1/8, gamma2=1/16, mu=0.058/14, p=[0,0.9])
    parameters['change_p'] = 31 
    writing(parameters)
    main(functions, file_name='scenario_A3_new')
    print(calculates_statistics('scenario_A3_new.txt'))

    # A4
    parameters['image_name'] = "scenario_A4_new.svg"
    functions = Parameters_Functions([[0.7676], [1, 0.2], [0.05]], [None, 31, None])
    parameters['parameters'] = change_values(parameters['parameters'], tau=1/3.2, sigma=1/2, alpha=0.4, delta=1/2, 
                                              epsilon=0, gamma1=1/8, gamma2=1/16, mu=0.058/14, p=[0,0.9])
    parameters['change_p'] = 31 
    writing(parameters)
    main(functions, file_name='scenario_A4_new')
    print(calculates_statistics('scenario_A4_new.txt'))

    # Scenario B

    # B1
    parameters['image_name'] = "scenario_B1_new.svg"
    functions = Parameters_Functions([[0.7676], [1, 0.5], [0.02]], [None, 35, None])
    parameters['parameters'] = change_values(parameters['parameters'], tau=1/3.2, sigma=1/2, alpha=0.4, delta=1/2, 
                                              epsilon=0, gamma1=1/8, gamma2=1/16, mu=0.034/14, p=[0,0.5])
    parameters['change_p'] = 35
    writing(parameters)
    main(functions, file_name='scenario_B1_new')
    print(calculates_statistics('scenario_B1_new.txt'))

    # B2
    parameters['image_name'] = "scenario_B2_new.svg"
    functions = Parameters_Functions([[0.7676], [1, 0.4], [0.02]], [None, 35, None])
    parameters['parameters'] = change_values(parameters['parameters'], tau=1/3.2, sigma=1/2, alpha=0.4, delta=1/2, 
                                              epsilon=0, gamma1=1/8, gamma2=1/16, mu=0.034/14, p=[0,0.65])
    parameters['change_p'] = 35
    writing(parameters)
    main(functions, file_name='scenario_B2_new')
    print(calculates_statistics('scenario_B2_new.txt'))

    # B3
    parameters['image_name'] = "scenario_B3_new.svg"
    functions = Parameters_Functions([[0.7676], [1, 0.3], [0.02]], [None, 35, None])
    parameters['parameters'] = change_values(parameters['parameters'], tau=1/3.2, sigma=1/2, alpha=0.4, delta=1/2, 
                                              epsilon=0, gamma1=1/8, gamma2=1/16, mu=0.034/14, p=[0,0.8])
    parameters['change_p'] = 35
    writing(parameters)
    main(functions, file_name='scenario_B3_new')
    print(calculates_statistics('scenario_B3_new.txt'))

    # B4
    parameters['image_name'] = "scenario_B4_new.svg"
    functions = Parameters_Functions([[0.7676], [1, 0.2], [0.02]], [None, 35, None])
    parameters['parameters'] = change_values(parameters['parameters'], tau=1/3.2, sigma=1/2, alpha=0.4, delta=1/2, 
                                              epsilon=0, gamma1=1/8, gamma2=1/16, mu=0.034/14, p=[0,0.9])
    parameters['change_p'] = 35
    writing(parameters)
    main(functions, file_name='scenario_B4_new')
    print(calculates_statistics('scenario_B4_new.txt'))

    # Scenario C 

    # C1
    parameters['image_name'] = "scenario_C1_new.svg"
    functions = Parameters_Functions([[0.7676], [1, 0.2], [0.02]], [None, 21, None])
    parameters['parameters'] = change_values(parameters['parameters'], tau=1/3.2, sigma=1/2, alpha=0.4, delta=1/2, 
                                              epsilon=0, gamma1=1/8, gamma2=1/16, mu=0.058/14, p=[0,0.9])
    parameters['change_p'] = 21
    writing(parameters)
    main(functions, file_name='scenario_C1_new')
    print(calculates_statistics('scenario_C1_new.txt'))

    # C2
    parameters['image_name'] = "scenario_C2_new.svg"
    functions = Parameters_Functions([[0.7676], [1, 0.2], [0.02]], [None, 49, None])
    parameters['parameters'] = change_values(parameters['parameters'], tau=1/3.2, sigma=1/2, alpha=0.4, delta=1/2, 
                                              epsilon=0, gamma1=1/8, gamma2=1/16, mu=0.058/14, p=[0,0.9])
    parameters['change_p'] = 49
    writing(parameters)
    main(functions, file_name='scenario_C2_new')
    print(calculates_statistics('scenario_C2_new.txt'))

    # Scenario D

    # D1
    parameters['image_name'] = "scenario_D1_new.svg"
    functions = Parameters_Functions([[0.7676], [1, 0.2], [0.1, 0.05]], [None, 50, 50])
    parameters['parameters'] = change_values(parameters['parameters'], tau=1/3.2, sigma=1/2, alpha=0.4, delta=1/2, 
                                              epsilon=0, gamma1=1/8, gamma2=1/16, mu=0.034/14, p=[0,0.8])
    parameters['change_p'] = 50
    writing(parameters)
    main(functions, file_name='scenario_D1_new')
    print(calculates_statistics('scenario_D1_new.txt'))

    # D2
    parameters['image_name'] = "scenario_D2_new.svg"
    functions = Parameters_Functions([[0.7676], [1, 0.2], [0.01, 0.1]], [None, 50, 50])
    parameters['parameters'] = change_values(parameters['parameters'], tau=1/3.2, sigma=1/2, alpha=0.4, delta=1/2, 
                                              epsilon=0, gamma1=1/8, gamma2=1/16, mu=0.034/14, p=[0,0.8])
    parameters['change_p'] = 50
    writing(parameters)
    main(functions, file_name='scenario_D2_new')
    print(calculates_statistics('scenario_D2_new.txt'))

    # Scenario E

    # E1
    parameters['image_name'] = "scenario_E1_new.svg"
    functions = Parameters_Functions([[0.7676], [1, 0.2], [0.0]], [None, 50, None])
    parameters['parameters'] = change_values(parameters['parameters'], tau=1/3.2, sigma=1/2, alpha=0.4, delta=1/2, 
                                              epsilon=0, gamma1=1/8, gamma2=1/16, mu=0.034/14, p=[0,0.8])
    parameters['change_p'] = 50
    writing(parameters)
    main(functions, file_name='scenario_E1_new')
    print(calculates_statistics('scenario_E1_new.txt'))

    # E2
    parameters['image_name'] = "scenario_E2_new.svg"
    functions = Parameters_Functions([[0.7676], [1, 0.2], [0.02]], [None, 50, None])
    parameters['parameters'] = change_values(parameters['parameters'], tau=1/3.2, sigma=1/2, alpha=0.4, delta=1/2, 
                                              epsilon=0, gamma1=1/8, gamma2=1/16, mu=0.034/14, p=[0,0.8])
    parameters['change_p'] = 50
    writing(parameters)
    main(functions, file_name='scenario_E2_new')
    print(calculates_statistics('scenario_E2_new.txt'))

    # E3
    parameters['image_name'] = "scenario_E3_new.svg"
    functions = Parameters_Functions([[0.7676], [1, 0.2], [0.05]], [None, 50, None])
    parameters['parameters'] = change_values(parameters['parameters'], tau=1/3.2, sigma=1/2, alpha=0.4, delta=1/2, 
                                              epsilon=0, gamma1=1/8, gamma2=1/16, mu=0.034/14, p=[0,0.8])
    parameters['change_p'] = 50
    writing(parameters)
    main(functions, file_name='scenario_E3_new')
    print(calculates_statistics('scenario_E3_new.txt'))

    # E4
    parameters['image_name'] = "scenario_E4_new.svg"
    functions = Parameters_Functions([[0.7676], [1, 0.2], [0.1]], [None, 50, None])
    parameters['parameters'] = change_values(parameters['parameters'], tau=1/3.2, sigma=1/2, alpha=0.4, delta=1/2, 
                                              epsilon=0, gamma1=1/8, gamma2=1/16, mu=0.034/14, p=[0,0.8])
    parameters['change_p'] = 50
    writing(parameters)
    main(functions, file_name='scenario_E4_new')
    print(calculates_statistics('scenario_E4_new.txt'))