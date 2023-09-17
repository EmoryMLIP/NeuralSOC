import torch
import torch.nn as nn
from torch.nn.functional import pad
from utils import normpdf
import matplotlib.pyplot as plt
from AbstractOCProblem import AbstractOCProblem

class QuadcopterProblem(AbstractOCProblem):
    """Definition of Quadcopter Problem"""
    def __init__(self):
        super().__init__()
        self.d = 12
        self.xtarget = torch.tensor([2.0,2.0,2.0,0,0,0,0,0,0,0,0,0]).reshape(1,12)
        self.mass = 1.0
        self.grav = 9.81
        self.sigma_const = 0.2
        self.t = 0.0
        self.T = 1.0

    def _get_name(self):
        return 'QuadcopterProblem'

    def __str__(self):
        return "TrajectoryProblem(d=%d, xtarget=[%1.2f,%1.2f,%1.2f], t=%1.2f, T=%1.2f, sigma=%1.2e)" %(self.d,self.xtarget[0,0],self.xtarget[0,1],self.xtarget[0,2],self.t,self.T,self.sigma_const)

    def x_init(self,nex):
        x0 = torch.Tensor([[-1.5,-1.5,-1.5]]) + torch.randn(nex,3)
        x0 = pad(x0, [0,self.d-3,0,0], value=0)
        return x0
    
    def ff(self, x):
        # angular function corresponding to problem
        # ang: [ψ,θ,ϕ]
        ang = x[:, 3:6]
        sinPsi   = torch.sin(ang[:, 0])
        sinTheta = torch.sin(ang[:, 1])
        sinPhi   = torch.sin(ang[:, 2])
        cosPsi   = torch.cos(ang[:, 0])
        cosTheta = torch.cos(ang[:, 1])
        cosPhi   = torch.cos(ang[:, 2])
        
        # f7 = sin(ψ) sin(ϕ) + cos(ψ) sin(θ) cos(ϕ)
        f7   = sinPsi * sinPhi + cosPsi * sinTheta * cosPhi
        # f8 = - cos(ψ) sin(ϕ) + sin(ψ) sin(θ) cos(ϕ)
        f8   = - cosPsi * sinPhi + sinPsi * sinTheta * cosPhi
        # f9 = cos(θ) cos(ϕ)
        f9   = cosTheta * cosPhi
        
        return f7 , f8 , f9
        
    def sigma(self,t,x):
        return self.sigma_const
        
    def sigma_mv(self,t,x,dw):
        sigma = self.sigma(t,x)
        return sigma*dw    
        
    def g(self, x):
        # terminal condition for value function
        res = x - self.xtarget
        G   = 0.5 * torch.norm(res, dim=1,keepdim=True)**2
        return 5000*G, 5000*res
        
    def f(self,s,z,u):
        f7c, f8c, f9c = self.ff(z)
        out = torch.cat((z[:, 6:], (u[:,[0]] / self.mass) * f7c.view(-1,1), (u[:,[0]] / self.mass) * f8c.view(-1,1),
                (u[:,[0]] / self.mass) * f9c.view(-1,1) - self.grav, u[:,1:]),dim=1)
        return out    
        
    def L(self, t, x, u):
        # running cost
        return 2 + torch.norm(u, dim=1, keepdim=True)**2
        
    def u_star(self,t,x,p):
        # define u in terms of p
        f7c, f8c, f9c = self.ff(x)
        u = torch.cat((1/(2*self.mass)*(f7c*p[:, 6]+f8c*p[:, 7]+f9c*p[:, 8]).view(-1, 1), p[:,[9]]/2, p[:,[10]]/2, p[:,[11]]/2),dim=1)
        return u   
       
    def Hamiltonian(self,t,x,p,M=None):    
        u =  self.u_star(t,x,p)  
        H = torch.sum(p*self.f(t,x,u), dim=1,keepdim=True) - self.L(t,x,u)
        # Here we assume M is always None for input
        gradpH = self.f(t,x,u)
        return H,gradpH
        
    def render(self, s,z,dw,Phi,sPath, nx=10):    
        # 3-d plot bounds
        xbounds = [-3.0, 3.0]
        ybounds = [-3.0, 3.0]
        zbounds = [-3.0, 3.0]
        
        Z = torch.stack(z,dim=-1)
        fig = plt.gcf()
        ax = fig.add_subplot(projection='3d')
        ax.set_title('Flight Path')
        
        ax.scatter(self.xtarget[0,0].cpu().numpy(), self.xtarget[0,1].cpu().numpy(), self.xtarget[0,2].cpu().numpy(), 
                   s=140, marker='x', c='r', label="target")
        for i in range(Z.shape[0]):
            ax.plot(Z[i, 0, :].view(-1).cpu().detach().numpy(), Z[i, 1, :].view(-1).cpu().detach().numpy(),
                    Z[i, 2, :].view(-1).cpu().detach().numpy(), 'o-')
        
        ax.view_init(10, -30)
        ax.set_xlim(*xbounds)
        ax.set_ylim(*ybounds)
        ax.set_zlim(*zbounds)
        
        fig.savefig(sPath, dpi=500, bbox_inches='tight')
        plt.show()
        plt.close('all')


if  __name__ == '__main__':
    prob = QuadcopterProblem()
    print(prob)
    nex = 10
    s = 0.3
    z = torch.randn((nex,prob.d))
    p = torch.randn_like(z)
    tt = prob.test_u_star(s,z,p)
    print(tt)

    prob.test_Hamiltonian(s,z,p)

    prob.test_g(z)
