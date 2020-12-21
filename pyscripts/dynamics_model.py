class Dynamics:
    
    ''' This class is used to describe the dynamics of the exposed model. '''
    
    def __init__(self, beta, r, tau, sigma, alpha, rho, delta, epsilon, gamma1, gamma2, mu): 
        """The init function organize the parameters. """        
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

    def derivatives(self, t, states): 
        """Calculate the system of derivatives"""
        Ef, Er, If, Ir, Af, Ar, Q, Sf, Sr, _, _, _ = tuple(states)
        
        diffEf = self.beta(t)*Sf*(If + Af + self.r(t)*(Ir + Ar) + self.eps*Q) - (self.rho(t)*self.delta*Ef + self.tau*Ef)
        
        diffEr = self.r(t)*self.beta(t)*Sr*(If + Af + self.r(t)*(Ir + Ar) + self.eps*Q) - (self.rho(t)*self.delta*Er + self.tau*Er)
        
        diffIf = self.tau*Ef - (self.sigma*If + self.rho(t)*If)
        
        diffIr = self.tau*Er - (self.sigma*Ir + self.rho(t)*Ir)
        
        diffAf = self.sigma*self.alpha*If - (self.rho(t)*Af + self.gamma1*Af)
        
        diffAr = self.sigma*self.alpha*Ir - (self.rho(t)*Ar + self.gamma1*Ar)
        
        diffSf = -(self.beta(t)*Sf*(If + Af + self.r(t)*(Ir + Ar) + self.eps*Q))
        
        diffSr = - (self.r(t)*self.beta(t)*Sr*(If + Af + self.r(t)*(Ir + Ar) + self.eps*Q))
        
        diffR = self.gamma1*(Af + Ar) + self.gamma2*Q
        
        diffT = self.sigma*(1 - self.alpha)*(If + Ir) + self.rho(t)*(self.delta*(Ef + Er) + If + Ir + Af + Ar)
        
        diffD = self.mu * Q
        
        diffQ = diffT - diffD - self.gamma2*Q

        return [diffEf, diffEr, diffIf, diffIr, diffAf, diffAr, diffQ,
                diffSf, diffSr, diffR, diffD, diffT]
    

class Parameters_Functions:

    def __init__(self): 
        pass 
        
    def pbeta(self,t):
        return 1.0

    def pr(self,t):
        if t<=31: return 1
        else: return 0.2
        
    def prho(self,t):
        return 0.1

