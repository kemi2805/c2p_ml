import numpy as np
import torch

from scipy.optimize import brentq

from .metric import metric

class conservs:

    def __init__(self, dens, stilde, tau):
        self.dens    = dens
        self.stilde  = stilde
        self.tau     = tau

    def __str__(self):
        return f"""Conservatives:
           dens:    {self.dens:.5g}
           stilde:  {self.stilde}
           tau:     {self.tau:.5g}
        """

class prims:

    def __init__(self,rho, vel, eps, press):
        self.rho   = rho
        self.vel   = vel
        self.eps   = eps
        self.press = press
        
    def __str__(self):
        return f"""Primitives:
           rho:    {self.rho:.5g}
           vel:    {self.vel}
           eps:    {self.eps:.5g}
           press:  {self.press:.5g}
        """
    def __sub__(self,other):
        return (torch.abs((self.rho-other.rho)/(1e-20+self.rho))+
                torch.abs((self.eps-other.eps)/(1e-20+self.eps))+
                torch.abs((self.vel[0]-other.vel[0])/(1e-20+self.vel[0]))+
                torch.abs((self.vel[1]-other.vel[1])/(1e-20+self.vel[1]))+
                torch.abs((self.vel[2]-other.vel[2])/(1e-20+self.vel[2])))/5
class c2p:

    def __init__(self,metric,conservs,eos):
        self.stildeU = metric.Raise(conservs.stilde)
        stildeNorm = np.sqrt(np.dot(self.stildeU,conservs.stilde))
        if ( stildeNorm > conservs.dens + conservs.tau ):
            fact = 0.9999 * (conservs.dens + conservs.tau)
            conservs.stilde *= fact/stildeNorm
            self.stildeU = metric.Raise(conservs.stilde)
            stildeNorm = fact
        self.conservs = conservs
        self.metric   = metric
        self.eos = eos
        self.q = conservs.tau / conservs.dens
        self.r = stildeNorm / conservs.dens
        self.k = self.r / ( 1 + self.q )
        self.t = conservs.dens / self.metric.sqrtg 
        
    def Wtilde(self,z):
        return np.sqrt(1 + z**2)

    def rhotilde(self,W):
        return self.t/W

    def epstilde(self, W, z):
        return W * self.q - self.r * z + z**2/(1+W)

    def atilde(self,rho,eps):
        press = self.eos.press__eps_rho(torch.tensor(eps),torch.tensor(rho)).item()
        return press / (rho * (1+eps))

    def htilde(self,z):
        W = self.Wtilde(z)
        rho = self.rhotilde(W)
        epsmin,epsmax = self.eos.eps_range__rho(torch.tensor(rho))
        epsmin = epsmin.item()
        eps = max(epsmin,min(epsmax,self.epstilde(W,z)))
        return ( 1 + eps ) * ( 1 + self.atilde(rho,eps))

    def invert(self, zeta=None):

        func = lambda z: z - self.r / self.htilde(z)
        zm = 0.5 * self.k / np.sqrt(1-(0.5*self.k)**2)
        zp = 1e-06 + self.k / np.sqrt(1-self.k**2)

        # Call brent on func
        if zeta is None:
            zeta, _ = brentq(func,zm,zp,xtol=1e-15,full_output=True)

        W = self.Wtilde(zeta)

        rho = self.t / W
        
        epsmin, epsmax = self.eos.eps_range__rho(rho)
        
        eps = min( epsmax, max( epsmin, self.epstilde(W,zeta)))
        press = self.eos.press__eps_rho(eps,rho)

        h = self.htilde(zeta)

        vel = self.stildeU / self.t / h / W

        return prims(rho,vel,eps,press), zeta

    
def get_W(metric,prims):
    v2 = metric.square_norm_upper(prims.vel)
    return 1./np.sqrt(1-v2)

def p2c(metric,prims):
    if prims.vel.ndim == 1:
        return p2c_impl(metric,prims,False)
    else:
        return p2c_impl(metric,prims,True)

def p2c_impl(metric,prims,batched):
    W = get_W(metric,prims)
    
    u0 = W / metric.alp 
    alp_sqrtgamma = metric.alp * metric.sqrtg
    
    dens = metric.sqrtg * W * prims.rho 
    
    one_over_alp2 = 1./metric.alp**2
    rho0_h = prims.rho * ( 1 + prims.eps ) + prims.press
    alp2_sqrtgamma = metric.alp**2 * metric.sqrtg
    g4uptt = - one_over_alp2
    
    Tuptt = rho0_h * u0**2 + prims.press * g4uptt
    
    tau = alp2_sqrtgamma * Tuptt - dens
    
    vD = metric.lower_index(prims.vel)
    stilde = metric.sqrtg * vD
            
    if batched:
        stilde *= rho0_h[:,np.newaxis] * W[:,np.newaxis]**2
    else:
        stilde *= rho0_h * W**2
    
    return conservs(dens,stilde,tau)

def undensitize(metric,cons):
    cons.dens /= metric.sqrtg
    cons.stilde /= metric.sqrtg
    cons.tau /= metric.sqrtg
    
def densitize(metric,cons):
    cons.dens *= metric.sqrtg
    cons.stilde *= metric.sqrtg
    cons.tau *= metric.sqrtg
