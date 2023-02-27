import torch
import torch.nn as nn
from torch.nn.functional import pad
import copy

def antiderivTanh(x): # activation function aka the antiderivative of tanh
    return torch.abs(x) + torch.log(1+torch.exp(-2.0*torch.abs(x)))

def derivTanh(x): # act'' aka the second derivative of the activation function antiderivTanh
    return 1 - torch.pow( torch.tanh(x) , 2 )

class ResNN(nn.Module):
    def __init__(self, d, m, nTh=2):
        """
            ResNet N portion of Phi
        :param d:   int, dimension of space input (expect inputs to be d+1 for space-time)
        :param m:   int, hidden dimension
        :param nTh: int, number of resNet layers , (number of theta layers)
        """
        super().__init__()

        if nTh < 2:
            print("nTh must be an integer >= 2")
            exit(1)

        self.d = d
        self.m = m
        self.nTh = nTh
        self.layers = nn.ModuleList([])
        self.layers.append(nn.Linear(d + 1, m, bias=True)) # opening layer
        self.layers.append(nn.Linear(m,m, bias=True)) # resnet layers
        for i in range(nTh-2):
            self.layers.append(copy.deepcopy(self.layers[1]))
        self.act = antiderivTanh
        self.dact = torch.tanh
        self.d2act = derivTanh
        self.h = 1.0 / (self.nTh-1) # step size for the ResNet

    def forward(self, x):
        """
            N(s;theta). the forward propogation of the ResNet
        :param x: tensor nex-by-d+1, inputs
        :return:  tensor nex-by-m,   outputs
        """
        x = self.act(self.layers[0].forward(x))
        for i in range(1,self.nTh):
            x = x + self.h * self.act(self.layers[i](x))

        return x



class Phi_OTflow(nn.Module):
    def __init__(self, nTh, m, d, r=10):
        """
            neural network approximating Phi (see Eq. (9) in our paper)

            Phi( x,t ) = w'*ResNet( [x;t]) + 0.5*[x' t] * A'A * [x;t] + b'*[x;t] + c

        :param nTh:  int, number of resNet layers , (number of theta layers)
        :param m:    int, hidden dimension
        :param d:    int, dimension of space input (expect inputs to be d+1 for space-time)
        :param r:    int, rank r for the A matrix
        :param alph: list, alpha values / weighted multipliers for the optimization problem
        """
        super().__init__()

        self.m    = m
        self.nTh  = nTh
        self.d    = d

        self.r = min(r,d+1) # if number of dimensions is smaller than default r, use that

        self.A  = nn.Parameter(torch.zeros(self.r, d+1) , requires_grad=True)
        self.A  = nn.init.xavier_uniform_(self.A)
        self.c  = nn.Linear( d+1  , 1  , bias=True)  # b'*[x;t] + c
        self.w  = nn.Linear( m    , 1  , bias=False)

        self.net = ResNN(d, m, nTh=nTh)

        # set initial values
        self.w.weight.data = torch.ones(self.w.weight.data.shape)
        self.c.weight.data = torch.zeros(self.c.weight.data.shape)
        self.c.bias.data   = torch.zeros(self.c.bias.data.shape)

    def __str__(self):
        return "%s(d=%d, m=%d, nTh=%d, r=%r, act=%s)" % (self._get_name(),self.d, self.m, self.nTh, self.r, self.net.act.__name__ )

    def forward(self, s,x,do_gradient=False, do_Laplacian=False):
        """ calculating Phi(s, theta)...not used in OT-Flow """
        y = pad(x,[0,1,0,0],value=s)
        symA = torch.matmul(torch.t(self.A), self.A)  # A'A
        Phic = self.w(self.net(y)) + 0.5 * torch.sum(torch.matmul(y, symA) * y, dim=1, keepdims=True) + self.c(y)

        if do_gradient is False and do_Laplacian is False:
            return Phic
        elif do_gradient is True and do_Laplacian is False:
            dPhiy = self.trHess(y,do_Laplacian=False)
            dPhidx = dPhiy[:, :-1]
            dPhidt = dPhiy[:, -1]
            return Phic, dPhidx, dPhidt.unsqueeze(1)
        else:
            dPhiy,LapPhix = self.trHess(y, do_Laplacian=True)
            dPhidx = dPhiy[:, :-1]
            dPhidt = dPhiy[:, -1]
            return Phic, dPhidx, dPhidt.unsqueeze(1), LapPhix


    def trHess(self,x,d=None, do_Laplacian=True):
        """
        compute gradient of Phi wrt x and trace(Hessian of Phi); see Eq. (11) and Eq. (13), respectively
        recomputes the forward propogation portions of Phi

        :param x: input data, torch Tensor nex-by-d
        :param justGrad: boolean, if True only return gradient, if False return (grad, trHess)
        :return: gradient , trace(hessian)    OR    just gradient
        """

        # code in E = eye(d+1,d) as index slicing instead of matrix multiplication
        # assumes specific N.act as the antiderivative of tanh

        N    = self.net
        m    = N.layers[0].weight.shape[0]
        nex  = x.shape[0] # number of examples in the batch
        if d is None:
            d    = x.shape[1]-1
        symA = torch.matmul(self.A.t(), self.A)

        u = [] # hold the u_0,u_1,...,u_M for the forward pass
        z = N.nTh*[None] # hold the z_0,z_1,...,z_M for the backward pass
        # preallocate z because we will store in the backward pass and we want the indices to match the paper

        # Forward of ResNet N and fill u
        opening     = N.layers[0].forward(x) # K_0 * S + b_0
        u.append(N.act(opening)) # u0
        feat = u[0]

        for i in range(1,N.nTh):
            feat = feat + N.h * N.act(N.layers[i](feat))
            u.append(feat)

        # going to be used more than once
        tanhopen = N.dact(opening) # act'( K_0 * S + b_0 )

        # compute gradient and fill z
        for i in range(N.nTh-1,0,-1): # work backwards, placing z_i in appropriate spot
            if i == N.nTh-1:
                term = self.w.weight.t()
            else:
                term = z[i+1]

            # z_i = z_{i+1} + h K_i' diag(...) z_{i+1}
            z[i] = term + N.h * torch.mm( N.layers[i].weight.t() , N.dact( N.layers[i].forward(u[i-1]) ).t() * term)

        # z_0 = K_0' diag(...) z_1
        z[0] = torch.mm( N.layers[0].weight.t() , tanhopen.t() * z[1] )
        grad = z[0] + (torch.mm(symA, x.t() ) + self.c.weight.t())

        if do_Laplacian is False:
            return grad.t()

        # -----------------
        # trace of Hessian
        #-----------------

        # t_0, the trace of the opening layer
        Kopen = N.layers[0].weight[:,0:d]    # indexed version of Kopen = torch.mm( N.layers[0].weight, E  )
        temp  = N.d2act(opening.t()) * z[1]
        trH  = torch.sum(temp.reshape(m, -1, nex) * torch.pow(Kopen.unsqueeze(2), 2), dim=(0, 1)) # trH = t_0

        # grad_s u_0 ^ T
        temp = tanhopen.t()   # act'( K_0 * S + b_0 )
        Jac  = Kopen.unsqueeze(2) * temp.unsqueeze(1) # K_0' * act'( K_0 * S + b_0 )
        # Jac is shape m by d by nex

        # t_i, trace of the resNet layers
        # KJ is the K_i^T * grad_s u_{i-1}^T
        for i in range(1,N.nTh):
            KJ  = torch.mm(N.layers[i].weight , Jac.reshape(m,-1) )
            KJ  = KJ.reshape(m,-1,nex)
            if i == N.nTh-1:
                term = self.w.weight.t()
            else:
                term = z[i+1]

            temp = N.layers[i].forward(u[i-1]).t() # (K_i * u_{i-1} + b_i)
            t_i = torch.sum(  ( N.d2act(temp) * term ).reshape(m,-1,nex)  *  torch.pow(KJ,2) ,  dim=(0, 1) )
            trH  = trH + N.h * t_i  # add t_i to the accumulate trace
            Jac = Jac + N.h * N.dact(temp).reshape(m, -1, nex) * KJ # update Jacobian

        return grad.t(), (trH + torch.trace(symA[0:d,0:d])).unsqueeze(1)
        # indexed version of: return grad.t() ,  trH + torch.trace( torch.mm( E.t() , torch.mm(  symA , E) ) )




if  __name__ == '__main__':

    d = 2
    nex = 2
    width = 8
    Phi = Phi_OTflow(nTh=2, m=width, d=d)
    print(Phi)

    print("Check input derivatives")
    x = torch.randn(nex,d,requires_grad=False)
    dx = torch.randn_like(x)
    Phic,gradPhi,dPhidt = Phi(0.0,x,do_gradient=True)
    dPhidx = torch.sum(gradPhi * dx, dim=1, keepdim=True)
    for k in range(10):
        h = 0.5 ** k
        Phit = Phi(h,x+h*dx,do_gradient=False)

        E0 = torch.norm(Phic - Phit)
        E1 = torch.norm(Phic + h*(dPhidx +dPhidt) - Phit)

        print("h=%1.2e\tE0=%1.2e\tE1=%1.2e" % (h, E0, E1))

    print("Check derivatives w.r.t. weights")

    Phic, gradPhi, dPhidt,LapPhi = Phi(0.0, x, do_gradient=True,do_Laplacian=True)
    # w = torch.zeros(1, 4); w[0, -1] = 1.0  # test dPhidt w.r.t. theta
    # w = torch.zeros(1, 4); w[0, 0] = 1.0  # test Phi w.r.t. theta
    w = torch.zeros(1, 5);
    w[0, 1:3] = 1.0  # test gradPhi w.r.t. theta
    w = torch.ones(1, 5);

    print(w)
    F = torch.sum(w*torch.cat((Phic,gradPhi,dPhidt,LapPhi),dim=1))
    # F = torch.sum(dphidy)
    F.backward()
    W0 = Phi.N.layers[0].weight.data.clone()
    W = Phi.N.layers[0].weight
    dW = torch.randn_like(W0)
    dFdW = torch.sum(dW * W.grad)
    for k in range(20):
        h = 0.5 ** k
        Phi.N.layers[0].weight.data = W0 + h*dW
        Phit, gradPhit, dPhidtt,LapPhit = Phi(0.0, x, do_gradient=True,do_Laplacian=True)
        Ft = torch.sum(w * torch.cat((Phit, gradPhit, dPhidtt,LapPhit), dim=1))
        # Ft = torch.sum(dPhidyt)
        E0 = torch.norm(F - Ft)
        E1 = torch.norm(F + h * dFdW - Ft)

        print("h=%1.2e\tE0=%1.2e\tE1=%1.2e" % (h, E0, E1))




