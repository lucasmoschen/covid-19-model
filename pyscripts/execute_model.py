#!/usr/bin/env python
# coding: utf-8

import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint 

import yaml
import os
import sys

#from dynamics_model import pbeta, pr, prho
from dynamics_model import Dynamics, Parameters_Functions

class States(Dynamics): 
    
    def __init__(self, beta, r, tau, sigma, alpha, rho, delta, epsilon, gamma1, gamma2, mu, 
                 initial,p,hmax):

        """The init function orhganize the parameters. """
        super().__init__(beta, r, tau, sigma, alpha, rho, delta, epsilon, gamma1, gamma2, mu)
        
        """ Initial Conditions"""
        self.y0 = [initial['E_0']*(1 - p), initial['E_0']*p, initial['I_0']*(1 - p), initial['I_0']*p, 
                   initial['A_0']*(1 - p), initial['A_0']*p, initial['Q_0'], 
                   initial['S_0']*(1 - p), initial['S_0']*p, initial['R_0'], initial['D_0'], initial['T_0']]
                   
        self.hmax = hmax

    def state_model(self, t0, tn):
        
        dynamics = lambda states, t: Dynamics.derivatives(self, t, states) 
    
        print("INFO - Calling the Integration method.")
        odesolver = odeint(func = dynamics, y0 = self.y0, t = np.linspace(t0,tn,tn-t0+1), 
                           hmax = self.hmax, full_output = 1)
        
        odesolver_var = odesolver[0].transpose()
        odesolver_info = odesolver[1]

        return odesolver_var, odesolver_info 

    def redistribute(self, y0): 
        """Redistribute the initial values."""
        S0 = (y0[7] + y0[8])
        E0 = (y0[0] + y0[1])
        A0 = (y0[4] + y0[5])
        I0 = (y0[2] + y0[3])
        
        return {'S_0': S0, 'E_0': E0, 'I_0': I0, 'A_0': A0, 'Q_0': y0[6], 
                'R_0': y0[9], 'D_0': y0[10], 'T_0': y0[11]}
    
def plotting(t,x, name):
    """ x is a dictionary with the name and a list with lists of the informations to plot"""
    plt.figure(figsize=(80,100))
    for i in range(len(x.keys())):
        plt.subplot(int(len(x.keys())/3) + 1,3,i+1)
        plt.plot(t, x[list(x.keys())[i]])
        plt.ticklabel_format(useOffset=False)
        plt.title(list(x.keys())[i])
    plt.savefig("../images/" + name)

def main(functions, file_name = False, info = False):

    print("INFO - Reading the parameters file. ")
    with open("../data/parameters.yaml") as f:
        data = yaml.load(f, Loader = yaml.FullLoader)
        par = data["parameters"]
        initial = data["initial"]
        Tf = data["Tf"]
        change_p = data['change_p']
        Image_Name = data["image_name"]
        hmax = data["hmax"]

    pbeta = functions.pbeta
    prho = functions.prho
    pr = functions.pr

    t_list = np.linspace(0,Tf,Tf + 1)

    ## Calculating $R_0$
    
    def R0Calc(t0, tn, p):

        r0_list = np.zeros(tn - t0 + 1)
        for t in range(t0, tn + 1):  
            # calculates de varphi quantity 
            varphi = pbeta(t)*par['tau']*(1 - (1 - pr(t)**2)*p)/((prho(t)*par['delta'] + par['tau'])*(par['sigma'] + prho(t)))
            r0 = 1/2*(varphi + np.sqrt(varphi**2 + (4*par['sigma']*par['alpha']/(prho(t) + par['gamma1']))*varphi))
            r0_list[t-t0] = r0

        return r0_list


    states = np.array([[] for i in range(12)])
    r0_list = []
    tn = -1
    for p in par['p']: 
        t0, tn = tn + 1, change_p
        covid = States(pbeta, pr, par["tau"], par["sigma"], par["alpha"], 
                    prho, par["delta"], par["epsilon"], par["gamma1"], par["gamma2"], par["mu"], 
                    initial, float(p), float(hmax))

        print("INFO - Calculating the States")
        odesolver, info  = covid.state_model(t0, tn)
        r0 = R0Calc(t0, tn, float(p))
        r0_list.extend(r0)
        
        states = np.hstack([states, odesolver])
        initial = covid.redistribute(states[:,-1])

        change_p = Tf

    Ef, Er, If, Ir, Af, Ar, Q, Sf, Sr, R, D, T = states

    variables = dict(zip(["t","Ef","Er","If","Ir","Af", "Ar", "Q","Sf","Sr","R","D","T", "R0"],
                         [t_list,Ef, Er, If, Ir, Af, Ar, Q, Sf, Sr, R, D, T, r0_list]))        

    if not file_name: 
        print("INFO - Saving the States variables. Please, input a name for the file name:")
        file_name = input()

    if not os.path.exists("../data/variables"):
        os.mkdir("../data/variables")

    with open("../data/variables/"+file_name+".txt", "w") as f:
        for i in variables.keys():
            f.write(i)
            for j in variables[i]:
                f.write(",")
                f.write(str(j))
            f.write("\n")
                
    variables.pop("t")
    
    if not info: 
        print("INFO - Information about the integration")
        while True:
            print("QUESTION - Do you wanna print the integration information? (y/n)")
            yn = input()
            if yn == "y":
                print("INFO - Printing")
                yn = True
            elif yn == "n":
                print("INFO - Not printing")
                yn = False
            break

        if yn: 
            print(info)
   
    plotting(t_list, variables, Image_Name)

if __name__ == "__main__":

    main(Parameters_Functions())