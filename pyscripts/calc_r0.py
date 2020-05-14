import sympy as sp
import numpy as np

class calc_R0:
    
    def __init__(self, X, infected, fX, R0 = None):
        """ This function allows to calculate R0 using the Jacobian and treating 
            the system in the DFE (Disease Free Equilibria). 
            
            X: list with states variables of the system 
            infected: list with the infected variables of the system"""
        self.X = X
        self.infected = infected
        self.fX = fX
        self.R0 = R0
        self.Jacobian = self.calc_Jacobian()
        
    def calc_Jacobian(self):
        
        Jacobian = sp.zeros(len(self.X), len(self.X))
        for i in range(len(self.X)):
            for j in range(len(self.X)):
                Jacobian[i, j] = sp.diff(self.fX[i], self.X[j])
                Jacobian[i, j] = Jacobian[i,j].subs(dict(zip(self.infected, np.zeros(len(self.infected)))))
        return Jacobian
        
    def calc_eig(self):
        
        eigenvalues = self.Jacobian.eigenvals()
        return list(eigenvalues.keys())
