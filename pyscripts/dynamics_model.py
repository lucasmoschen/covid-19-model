class Dynamics:
    
    ''' This class is used to describe the dynamics of the exposed model. '''
    
    def __init__(self, beta, r, tau, sigma, alpha, rho, delta, epsilon, gamma1, gamma2, mu): 
        """The init function orhganize the parameters. """        
        self.beta = beta #function
        self.r = r       #function
        self.tau = tau
        self.sigma = sigma
        self.alpha = alpha 
        self.rho = rho  #function
        self.delta = delta
        self.eps = epsilon
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.mu = mu
        
    def diffEf(self, t, states):
        """Calculate the dynamics of the exposed and free to move people. """
        Ef, Er, If, Ir, Af, Ar, Q, Sf, Sr, R, D, T = tuple(states)
        increase = self.beta(t)*Sf*(If + Af + self.r(t)*(Ir + Ar) + self.eps*Q)
        decrease = self.rho(t)*self.delta*Ef + self.tau*Ef
        return increase - decrease
    
    def diffEr(self, t, states):
        """Calculate the dynamics of the exposed, in r-isolation. """
        Ef, Er, If, Ir, Af, Ar, Q, Sf, Sr, R, D, T = tuple(states)
        increase = self.r(t)*self.beta(t)*Sr*(If + Af + self.r(t)*(Ir + Ar) + self.eps*Q)
        decrease = self.rho(t)*self.delta*Er + self.tau*Er
        return increase - decrease
        
    def diffIf(self, t, states):
        """Calculate the dynamics of the asymptomatic infected, but free to move people. """
        Ef, Er, If, Ir, Af, Ar, Q, Sf, Sr, R, D, T = tuple(states)
        increase = self.tau*Ef
        decrease = self.sigma*If + self.rho(t)*If
        return increase - decrease
    
    def diffIr(self, t, states):
        """Calculate the dynamics of the asymptomatic infected, in r-isolation. """
        Ef, Er, If, Ir, Af, Ar, Q, Sf, Sr, R, D, T =  tuple(states)
        increase = self.tau*Er
        decrease = self.sigma*Ir + self.rho(t)*Ir
        return increase - decrease
    
    def diffAf(self, t, states):
        """Calculate the dynamics of the asymptomatic infected, but free to move people. """
        Ef, Er, If, Ir, Af, Ar, Q, Sf, Sr, R, D, T = tuple(states)
        increase = self.sigma*self.alpha*If 
        decrease = self.rho(t)*Af + self.gamma1*Af
        return increase - decrease
    
    def diffAr(self, t, states):
        """Calculate the dynamics of the asymptomatic infected, in r-isolation. """
        Ef, Er, If, Ir, Af, Ar, Q, Sf, Sr, R, D, T =  tuple(states)
        increase = self.sigma*self.alpha*Ir
        decrease = self.rho(t)*Ar + self.gamma1*Ar
        return increase - decrease
    
    def diffQ(self,t, states):
        """Calculate the dynamics of the infected and positive tested, no moving. """
        Ef, Er, If, Ir, Af, Ar, Q, Sf, Sr, R, D, T =  tuple(states)
        increase = self.sigma*(1 - self.alpha)*(If + Ir) + self.rho(t)*(self.delta*(Ef + Er) + If + Ir + Af + Ar)
        decrease = self.gamma2*Q + self.mu*Q
        return increase - decrease
    
    def diffSf(self, t, states):
        """Calculate the dynamics of susceptible, free moving. """
        Ef, Er, If, Ir, Af, Ar, Q, Sf, Sr, R, D, T = tuple(states)
        increase = 0
        decrease = self.beta(t)*Sf*(If + Af + self.r(t)*(Ir + Ar) + self.eps*Q)
        return increase - decrease

    def diffSr(self, t, states):
        """Calculate the dynamics of susceptible, in volunteer r-isolation. """
        Ef, Er, If, Ir, Af, Ar, Q, Sf, Sr, R, D, T =  tuple(states)
        increase = 0
        decrease = self.r(t)*self.beta(t)*Sr*(If + Af + self.r(t)*(Ir + Ar) + self.eps*Q)
        return increase - decrease
    
    def diffR(self, t, states):
        """Calculate the dynamics of recovered or immune. """
        Ef, Er, If, Ir, Af, Ar, Q, Sf, Sr, R, D, T =  tuple(states)
        increase = self.gamma1*(Af + Ar) + self.gamma2*Q
        decrease = 0
        return increase - decrease
    
    def diffD(self, t, states):
        Ef, Er, If, Ir, Af, Ar, Q, Sf, Sr, R, D, T =  tuple(states)
        increase = self.mu*Q
        decrease = 0
        return increase - decrease
    
    def diffT(self, t, states):
        """Denote the total number of positive tests until time t"""
        Ef, Er, If, Ir, Af, Ar, Q, Sf, Sr, R, D, T =  tuple(states)
        increase = self.sigma*(1 - self.alpha)*(If + Ir) + self.rho(t)*(self.delta*(Ef + Er) + If + Ir + Af + Ar)
        decrease = 0
        return increase - decrease      
        
def pbeta(t):
    return 0.7676


def pr(t):
    if t<50: return 1
    else: return 0.2
    

def prho(t):
    return 0.1
