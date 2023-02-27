import torch
import numpy as np

class PMPSampler(torch.nn.Module):

    def __init__(self,Phi,prob,t,T,nt):
        super().__init__()
        self.t=t
        self.T=T
        self.nt=nt
        self.Phi = Phi
        self.prob = prob

    def __str__(self):
        return "%s(t=%1.2f, T=%1.2f, nt=%d)" % (self._get_name(), self.t, self.T, self.nt)


    def forward(self, x, t=None, T=None, nt=None):

        if t is None: t= self.t
        if T is None: T = self.T
        if nt is None: nt = self.nt

        s = torch.linspace(t, T, nt + 1)
        z = [x]
        Phizi, gradPhizi, _ = self.Phi(s[0], z[0], do_gradient=True)
        Phiz = [Phizi]
        gradPhiz = [gradPhizi]
        dw = [torch.randn_like(x)*np.sqrt(s[1]-s[0])]

        H, gradpH = self.prob.Hamiltonian(s[0], z[0], -gradPhiz[0], None)

        for i in range(nt):
            ds = s[i+1]-s[i]
            z.append(z[i] + gradpH * ds + self.prob.sigma_mv(s[i],z[i], dw[i]))

            Phizi, gradPhizi, _ = self.Phi(s[i+1], z[i+1], do_gradient=True)
            Phiz.append(Phizi); gradPhiz.append(gradPhizi)

            H, gradpH = self.prob.Hamiltonian(s[i+1], z[i+1], -gradPhiz[i+1], None)
            dw.append( torch.randn_like(x)*np.sqrt(ds))

        return s, z, dw, Phiz, gradPhiz



class RandomWalkSampler(torch.nn.Module):

    def __init__(self,prob,t,T,nt):
        super().__init__()
        self.t=t
        self.T=T
        self.nt=nt
        self.prob=prob

    def __str__(self):
        return "%s(t=%1.2f, T=%1.2f, nt=%d)" % (self._get_name(), self.t, self.T, self.nt)

    def forward(self, x, t=None, T=None, nt=None):

        if t is None: t= self.t
        if T is None: T = self.T
        if nt is None: nt = self.nt

        s = torch.linspace(t, T, nt + 1)
        z = [x]
        dw = [torch.randn_like(x)*np.sqrt(s[1]-s[0])]
        for i in range(nt):
            ds = s[i+1]-s[i]
            z.append(z[i] + self.prob.sigma_mv(s[i],z[i], dw[i]))
            dw.append( torch.randn_like(x)*np.sqrt(ds))

        return s, z, dw, None,None