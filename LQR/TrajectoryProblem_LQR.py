import torch
import torch.nn as nn
from torch.nn.functional import pad
from utils import normpdf
import matplotlib.pyplot as plt
from AbstractOCProblem import AbstractOCProblem

class TrajectoryProblem(AbstractOCProblem):
    """Definition of Trajectory Problem"""
    def __init__(self):
        super().__init__()
        self.d = 2
        self.xtarget = torch.tensor([0.0,0.0]).reshape(1,2)
        self.mu = torch.zeros(1,self.d)
        self.cov = 0.4 * torch.ones(1,self.d)
        self.sigma_const = 0.5
        self.t = 0.0
        self.T = 1.0

    def _get_name(self):
        return 'TrajectoryProblem'

    def __str__(self):
        return "TrajectoryProblem(d=%d, xtarget=[%1.2f,%1.2f], t=%1.2f, T=%1.2f, sigma=%1.2e)" %(self.d,self.xtarget[0,0],self.xtarget[0,1],self.t,self.T,self.sigma_const)

    def x_init(self,nex):
        return -3.0 * torch.ones(nex,2) + 1.0*torch.randn(nex,2)

    def f(self,s,z,u):
        return u

    def sigma(self,t,x):
        return self.sigma_const

    def sigma_mv(self,t,x,dw):
        sigma = self.sigma(t,x)
        return sigma*dw


    def L(self, t, x, u):
        #
        # need x(not including t) and grad u, here indicate by p
        '''edit'''
        return 0.5 * torch.norm(u, dim=1, keepdim=True)**2 # + 50.0 * normpdf(x, mu=self.mu, cov=self.cov)

    def g(self, x):
        # terminal condition for value function
        res = x - self.xtarget
        G   = 0.5 * torch.norm(res, dim=1,keepdim=True)**2
        return 100*G, 100*res

    def u_star(self,t,x,p):
        return p

    def Hamiltonian(self,t,x,p,M=None):
        u =  self.u_star(t,x,p)  # minimizer of calH
        H =  0.5 * torch.norm(u, dim=1,keepdim=True)**2 # - 50.0 * normpdf(x, mu=self.mu, cov=self.cov)
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

    def render(self, s,z,dw,Phi,sPath, nx=100):

        x = torch.linspace(-2.0, 2.0, steps=nx)
        y = torch.linspace(-2.0, 2.0, steps=nx)
        X, Y = torch.meshgrid(x, y)
        XY = torch.cat((X.reshape(-1, 1), Y.reshape(-1, 1)), 1)
        Q = 0. * normpdf(XY, mu=self.mu, cov=self.cov)

        Z = torch.cat(z,dim=1)
        fig = plt.gcf()
        ax = fig.add_subplot(231)
        ax.imshow(Q.view(nx,nx), cmap='hot',origin='lower',extent=(-2.0,2.0,-2.0,2.0))
        ax.plot(self.xtarget[0,0], self.xtarget[0,1], 'ob')
        for i in range(Z.shape[0]):
            ax.plot(Z[i, 0::2].detach().numpy(), Z[i, 1::2].detach().numpy(), '-o')

        ax.set_xlim(-2., 2.)
        ax.set_ylim(-2., 2.)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title('trajectories')

        ax = fig.add_subplot(232)
        Phi0 = Phi(self.t,XY)
        im = ax.imshow(Phi0.detach().view(nx, nx), cmap='hot', origin='lower', extent=(-2.0, 2.0, -2.0, 2.0))
        fig.colorbar(im, ax=ax)
        ax.plot(self.xtarget[0, 0], self.xtarget[0, 1], 'ob')
        ax.set_title('Phi(0)')

        ax = fig.add_subplot(233)
        PhiT = Phi(self.T,XY)
        im = ax.imshow(PhiT.detach().view(nx, nx), cmap='hot', origin='lower', extent=(-2.0, 2.0, -2.0, 2.0))
        fig.colorbar(im, ax=ax)
        ax.plot(self.xtarget[0, 0], self.xtarget[0, 1], 'ob')
        ax.set_title('Phi(T)')

        ax = fig.add_subplot(234)
        g = self.g(XY)[0]
        im = ax.imshow(g.detach().view(nx, nx), cmap='hot', origin='lower', extent=(-2.0, 2.0, -2.0, 2.0))
        ax.plot(self.xtarget[0, 0], self.xtarget[0, 1], 'ob')
        fig.colorbar(im, ax=ax)
        ax.set_title('g')
        
        ax = fig.add_subplot(235)
        _, gradPhi0, _ = Phi(self.t,XY, do_gradient=True)
        ax.quiver(XY.detach().cpu().numpy()[::31, 0], XY.detach().cpu().numpy()[::31, 1], -gradPhi0.detach().cpu().numpy()[::31, 0], -gradPhi0.detach().cpu().numpy()[::31, 1])  # arrow at every 11th pt
        ax.set_title(r'$-\nabla_z \Phi (z, s=0)$')
        ax.set_aspect('equal')
        
        ax = fig.add_subplot(236)
        _, gradPhi0, _ = Phi(self.t+0.5*(self.T-self.t),XY, do_gradient=True)
        ax.quiver(XY.detach().cpu().numpy()[::31, 0], XY.detach().cpu().numpy()[::31, 1], -gradPhi0.detach().cpu().numpy()[::31, 0], -gradPhi0.detach().cpu().numpy()[::31, 1])  # arrow at every 11th pt
        ax.set_title(r'$-\nabla_z \Phi (z, s=0.5)$')
        ax.set_aspect('equal')


        fig.savefig(sPath, dpi=300)

        plt.show()
        plt.close('all')

class TrajectoryProblem2(TrajectoryProblem):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return "TrajectoryProblem(d=%d, xtarget=[%1.2f,%1.2f], t=%1.2f, T=%1.2f, sigma=%1.2e)" %(self.d,self.xtarget[0,0],self.xtarget[0,1],self.t,self.T,self.sigma_const)


    def sigma(self, t, x):
        '''
        v1 = x / torch.norm(x, dim=1, keepdim=True)
        v2 = torch.cat((v1[:, 1].unsqueeze(1), -v1[:, 0].unsqueeze(1)), dim=1)
        sigma = 1e-3*self.sigma_const * v1.unsqueeze(1) * v1.unsqueeze(2) + self.sigma_const * v2.unsqueeze(1) * v2.unsqueeze(2)
        '''
        sigma = torch.tensor([0.4,0.0,0.0,0.4]).reshape(2,2)
        return sigma

    def sigma_mv(self, t, x, dw):
        sigma = self.sigma(t, x)

        # return torch.sum(sigma * dw.unsqueeze(1), dim=2)
        return (sigma @ dw.unsqueeze(2)).reshape(dw.shape)

    def tr_sigma2_M(self, t,x,M):
        '''
        sigsigthess = self.sigma(t,x)*torch.transpose(self.sigma(t,x),1,2)*M
        tr_sigma2_M = 0.5 * torch.sum(
            sigsigthess * torch.eye(x.shape[1]).view(1, x.shape[1], x.shape[1]),
            dim=(1, 2)).unsqueeze(1)
        '''
        sigsigthess = self.sigma(t,x) @ self.sigma(t,x).t() @ M
        tr_sigma2_M = 0.5 * torch.sum(
            sigsigthess * torch.eye(x.shape[1]).view(1, x.shape[1], x.shape[1]),
            dim=(1, 2)).unsqueeze(1)    
            
        return tr_sigma2_M


if  __name__ == '__main__':
    prob = TrajectoryProblem2()
    print(prob)
    nex = 10
    s = 0.3
    z = torch.randn((nex,prob.d))
    p = torch.randn_like(z)
    tt = prob.test_u_star(s,z,p)
    print(tt)

    prob.test_Hamiltonian(s,z,p)

    prob.test_g(z)