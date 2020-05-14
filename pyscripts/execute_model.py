#!/usr/bin/env python
# coding: utf-8

# # COVID-19 Model 

# It's assumed the people are indicated to do a quarantine, reducing their contact to $100r$ of their usual contacts.

# Consider some parameters: 

# | Parameter | Explanation |
# |-----| ----------------------------------------|
# | $p$ | fraction of population in $r$-isolation |
# | $\tau$ | inverse of the latent time |
# | $\sigma$ | inverse of time before clear symptoms such to ensure isolation or testing of the subject |
# | $\theta$ | $\tau + \sigma$ inverse of mean incubation time | 
# | $\alpha$ | asymptomatic rate |
# | $\delta$ | probability of positive test in compartment $E$ |
# | $\gamma_1$ | recovery rate for asymptomatic or mild symptoms|
# | $\gamma_2$ | recovery rate for $Q$ |
# | $\mu$ | mortality rate of infected symptomatic |
# | $\beta(t)$ |contact rate among people free to move |
# | $\rho(t)$ | proportion of tests done in no infected with severe symptoms |
# | $r(t)$ | reduction coefficient of contact rate |

# ## SEIR with Q model

# Consider the state $X = (E_f, E_r, I_f, I_r, A_f, A_r, Q, S_f, S_r, R, D)$, where

# | State | Explanation |
# | ----- | ---------------------------------------------------- |
# | $E_f$ | exposed, not in isolation, not contagious |
# | $E_r$ | exposed, in volunteer $r$-isolation, not contagious |
# | $I_f$ | asymptomatic infected, not in isolation | 
# | $I_r$ | asymptpmatic infected, in volunteer $r$-isolation |
# | $A_f$ | Asymptomatic and contagious, not in isolation |
# | $A_r$ | Asymptomatic and contagious, r-isolation |
# | $Q$ | infected and tested positive, in enforced quarantine | 
# | $S_f$ | susceptible not in isolation |
# | $S_q$ | susceptible in volunteer $r$-isolation |
# | $R$ | recovered and immune |
# | $D$ | deaths |

# The dynamics is described as follows:

# $$
#     \begin{array}{l}
#     \dot{E}_f = \beta(t) S_f [I_f + A_f + r(t)(I_r + A_r) + \epsilon Q] - \rho(t)\delta E_f - \tau E_f \\[0.5ex]
#     \dot{E}_r = r(t) \beta(t) S_r [I_f + A_f + r(t)(I_r + A_r) + \epsilon Q] - \rho(t)\delta E_r - \tau E_r \\[0.5ex]
#     \dot{I}_f = \tau E_f - \sigma I_f - \rho(t)I_f \\[0.5ex]
#     \dot{I}_r = \tau E_r - \sigma I_r - \rho(t)I_r  \\[0.5ex]
#     \dot{A}_f = \sigma\alpha I_f - \rho(t)A_f - \gamma_1 A_f \\[0.5ex]
#     \dot{A}_r = \sigma\alpha I_r - \rho(t)A_r - \gamma_1 A_r \\[0.5ex]
#     \dot{Q} = \sigma (1-\alpha) [I_f + I_r] + \rho(t)(\delta(E_f + E_r) + I_f + I_r + A_f + A_r) - \gamma_2 Q - \mu Q \\[0.5ex]
#     \dot{S}_f = -\beta(t)S_f [I_f + A_f + r(t)(I_r + A_r) + \epsilon Q] \\[0.5ex]
#     \dot{S}_r = -r(t)\beta(t)S_r [I_f + A_f + r(t)(I_r + A_r) + \epsilon Q] \\[0.5ex]
#     \dot{R} = \gamma_1 (A_f + A_r) + \gamma_2 Q \\[0.5ex]
#     \dot{D} = \mu Q
#     \end{array}
# $$
# #### Importing modules

import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint 

import yaml
import os
import sys

from dynamics_model import pbeta, pr, prho
from dynamics_model import Dynamics
from calc_r0 import calc_R0

class States(Dynamics): 
    
    def __init__(self, beta, r, tau, sigma, alpha, rho, delta, epsilon, gamma1, gamma2, mu, 
                 initial,hmax):

        """The init function orhganize the parameters. """
        self.beta = beta 
        self.r = r
        self.tau = tau
        self.sigma = sigma
        self.alpha = alpha 
        self.rho = rho
        self.delta = delta 
        self.eps = epsilon
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.mu = mu
        
        """ Initial Conditions"""
        self.y0 = [initial['Ef_0'], initial['Er_0'], initial['If_0'], initial['Ir_0'], initial['Af_0'], initial['Ar_0'], 
                   initial['Q_0'], initial['Sf_0'], initial['Sr_0'], initial['R_0'], initial['D_0'], initial['T_0']]
                   
        self.hmax = hmax

    def state_model3(self, T):
        
        def dynamics(states, t): 
            
            fun = [Dynamics.diffEf(self,t,states), Dynamics.diffEr(self,t,states), Dynamics.diffIf(self,t,states), Dynamics.diffIr(self,t,states), 
                   Dynamics.diffAf(self,t,states), Dynamics.diffAr(self,t,states), Dynamics.diffQ(self,t,states), Dynamics.diffSf(self,t,states), 
                   Dynamics.diffSr(self,t,states), Dynamics.diffR(self,t,states), Dynamics.diffD(self,t,states), Dynamics.diffT(self,t,states)]
          
            return fun
    
        print("INFO - Calling the Integration method method")
        odesolver = odeint(func = dynamics, y0 = self.y0, t = np.linspace(0,T,T+1), 
                           hmax = self.hmax, full_output=1)
        
        odesolver_var = odesolver[0].transpose()
        odesolver_info = odesolver[1]
        
        Ef, Er, If, Ir, Af, Ar, Q, Sf, Sr, R, D, T = odesolver_var
        return Ef, Er, If, Ir, Af, Ar, Q, Sf, Sr, R, D, T, odesolver_info 
    
def plotting(t,x, name):
    """ x is a dictionary with the name and a list with lists of the informations to plot"""
    plt.figure(figsize=(80,100))
    for i in range(len(x.keys())):
        plt.subplot(int(len(x.keys())/3) + 1,3,i+1)
        plt.plot(t, x[list(x.keys())[i]])
        plt.ticklabel_format(useOffset=False)
        plt.title(list(x.keys())[i])
    plt.savefig("../images/" + name)

if __name__ == "__main__":

    print("INFO - Reading the parameters file. ")
    with open("../data/parameters.yaml") as f:
        data = yaml.load(f, Loader = yaml.FullLoader)
        par = data["parameters"]
        initial = data["initial"]
        Tf = data["Tf"]
        Image_Name = data["image_name"]
        hmax = data["hmax"]

    ## Calculating $R_0$
    
    def R0Calc(model, t_list):

        if os.path.exists("../data/r0.txt"):
            print("INFO - Reading the file with R0 already calculated")
            with open("../data/r0.txt", 'r') as f:
                line = f.readline()
                line = f.readline()
                
                t, p, tau, sigma, alpha, delta, epsilon, gamma1, gamma2, mu = sp.symbols("t p tau sigma alpha delta epsilon gamma1 gamma2 mu")
                beta, r, rho = sp.Function("beta"), sp.Function("r"), sp.Function("rho")
                
                param_r0 = par.copy()
                param_r0.update(zip(param_r0.keys(), [p, tau, sigma, alpha, delta, epsilon, gamma1, gamma2, mu]))
                param_r0['beta'] = beta
                param_r0['r'] = r
                param_r0['rho'] = rho

                R0 = parse_expr(line, local_dict = param_r0)
                r0_numerical = R0.subs(par)
                r0_list = [r0_numerical.subs({rho(t): prho(i), beta(t): pbeta(i), r(t): pr(i)}) for i in t_list]
            
        else: 
            print("No file R0. Treating it 0.")
            r0_list = np.zeros(len(t_list))

        return r0_list

    t_list = np.linspace(0,Tf,Tf + 1)
         
    covid = States(pbeta, pr, par["tau"], par["sigma"], par["alpha"], 
                   prho, par["delta"], par["epsilon"], par["gamma1"], par["gamma2"], par["mu"], 
                   initial, float(hmax))

    print("INFO - Calculating the States")
    Ef, Er, If, Ir, Af, Ar, Q, Sf, Sr, R, D, T, info  = covid.state_model3(Tf)
    r0_list = R0Calc(3, t_list)
    variables = dict(zip(["t","Ef","Er","If","Ir","Af", "Ar", "Q","Sf","Sr","R","D","T", "R0"],
            [t_list,Ef, Er, If, Ir, Af, Ar, Q, Sf, Sr, R, D, T, r0_list]))        

    print("INFO - Saving the States variables. Please, input a name for the file name:")
    name = input()

    if not os.path.exists("../data/variables"):
        os.mkdir("../data/variables")

    with open("../data/variables/"+name+".txt", "w") as f:
        for i in variables.keys():
            f.write(i)
            for j in variables[i]:
                f.write(",")
                f.write(str(j))
            f.write("\n")
                
    variables.pop("t")

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
    
    plt.show()