import torch
import numpy as np
import torch.nn as nn
from torch.nn.functional import pad
from utils import normpdf
import matplotlib.pyplot as plt
from AbstractOCProblem import AbstractOCProblem

class BenchmarkProblem(AbstractOCProblem):
    """Definition of 100-D Benchmark Problem"""
    def __init__(self, type=1):
        super().__init__()
        self.d = 100
        if type != 1:
            self.xtarget = 3*torch.ones(1,self.d)
            self.term_weight = 1000.0
            self.sigma_const = 0.4*np.sqrt(2.0)
        else:
            self.xtarget = torch.zeros(1,self.d)
            self.term_weight = 1.0
            self.sigma_const = np.sqrt(2.0)
        self.t = 0.0
        self.T = 1.0


    def x_init(self,nex):
        return torch.zeros(nex,self.d)

    def f(self,s,z,u):
        return 2.0*u

    def sigma(self,t,x):
        return self.sigma_const

    def sigma_mv(self,t,x,dw):
        sigma = self.sigma(t,x)
        return sigma*dw


    def L(self, t, x, u):
        #
        # need x(not including t) and grad u, here indicate by p
        '''edit'''
        return  torch.norm(u, dim=1, keepdim=True)**2

    def g(self, x):
        # terminal condition for value function
        if x.dim() == 3:
            res = x-self.xtarget.unsqueeze(2)
        else:
            res = x-self.xtarget
        # res = x
        G = self.term_weight*torch.log((1 + torch.norm(res, dim=1, keepdim=True)**2) / 2)
        dG = self.term_weight*2*res/(1 + torch.norm(res, dim=1, keepdim=True)**2)
        return G, dG

    def u_star(self,t,x,p):
        return p

    def Hamiltonian(self,t,x,p,M=None):
        u =  self.u_star(t,x,p)  # minimizer of calH
        H =  torch.sum(p*self.f(t,x,u),dim=1,keepdim=True) - self.L(t, x, u) # 2*u**2 -  u**2
        if M is not None:
            sigma = prob.sigma(t,x)
            if torch.is_tensor(sigma):
                if M.dim()==2 and M.shape[1]==1:
                    # assume M is Laplacian of Phi
                    H = H + 0.5*self.tr_sigma2_M(t,x,M)
            else:
                if M.dim() == 2 and M.shape[1] == 1:
                    # assume M is Laplacian of Phi
                    H = H + (sigma**2)/2 * M
                else:
                    LapPhi = torch.sum(M * torch.eye(M.shape[1]).unsqueeze(0),dim=(1,2)).unsqueeze(1)
                    H = H + (sigma**2)/2 * LapPhi




        gradpH = self.f(t,x,u)
        return H,gradpH
    def Phi_true(self,s,z,num_sample=10000):
        return -torch.log(torch.mean(torch.exp(-self.g(z.unsqueeze(2) + self.sigma(s,z)*np.sqrt(self.T-s)*torch.randn(z.shape[0],self.d,num_sample))[0]),dim=2))

        
    def render(self, s,z,dw,Phi,sPath):

        nex = z[0].shape[0]
        Phic = np.zeros((nex,s.shape[0]))
        Phit = np.zeros((nex,s.shape[0]))
        for k in range(s.shape[0]):
            Phic[:,k] = Phi(s[k],z[k]).squeeze(1).detach().numpy()
            Phit[:,k] = self.Phi_true(s[k],z[k]).squeeze(1).detach().numpy()
            
        fig = plt.figure()
        plt.subplot(1,2,1)
        for k in range(nex):
            plt.plot(s, Phic[k], "-r")
            plt.plot(s, Phit[k], "-b")
        plt.legend(("estimate","true"))

        plt.subplot(1, 2, 2)
        mErr = np.mean(np.abs(Phic-Phit),axis=0)
        plt.plot(s, mErr, "-r")

        fig.savefig(sPath, dpi=300)

        plt.show()
        plt.close('all')


if  __name__ == '__main__':
    prob = BenchmarkProblem()
    nex = 10
    s = 0.3
    z = torch.randn((nex,prob.d))
    p = torch.randn_like(z)
    tt = prob.test_u_star(s,z,p)
    print(tt)

    prob.test_Hamiltonian(s,z,p)

    prob.test_g(z)
