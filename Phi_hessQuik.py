import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.nn.functional import pad

class Phi_hessQuik(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self,s,x,do_gradient=False, do_Laplacian=False, do_Hessian=False):
        y = pad(x,[0,1,0,0],value=s)
        if do_gradient is False and do_Laplacian is False and do_Hessian is False:
            return self.net(y)[0]
        elif do_gradient is True and do_Laplacian is False and do_Hessian is False: # compute gradient in backward mode
            Phic, dPhiy, _ = self.net(y, do_gradient=True, do_Hessian=False)
            dPhidx = dPhiy[:, :-1].squeeze(2)
            dPhidt = dPhiy[:, -1]
            return Phic, dPhidx, dPhidt
        else:
            Phic, dPhiy, d2Phiy = self.net(y, do_gradient=True, do_Hessian=True)
            dPhidx = dPhiy[:,:-1]
            dPhidt = dPhiy[:,-1]
            if do_Hessian is True:
                return Phic, dPhidx.squeeze(2), dPhidt, d2Phiy[:,:-1,:-1].squeeze(3)
            else:
                # LapPhi = torch.sum(d2Phiy[:-1, :-1].unsqueeze(3) * torch.eye(x.shape[1]).view(1,x.shape[1],x.shape[1]), dim=(1,2)).unsqueeze(1)
                LapPhi = torch.sum(
                    d2Phiy[:,:-1, :-1].squeeze(3) * torch.eye(x.shape[1]).view(1, x.shape[1], x.shape[1]),
                    dim=(1, 2)).unsqueeze(1)
                return Phic, dPhidx.squeeze(2), dPhidt, LapPhi

if  __name__ == '__main__':
    import hessQuik.activations as act
    import hessQuik.layers as lay
    import hessQuik.networks as net

    d = 2
    nex = 2
    width = [8]

    net = net.NN(
        lay.singleLayer(d+1, width[0],act.tanhActivation()),
        lay.singleLayer(width[-1], 1, act.identityActivation())
    )

    Phi = Phi_hessQuik(net)
    print("Check input derivatives")
    x = torch.randn(nex,d,requires_grad=False)
    dx = torch.randn_like(x)
    Phic,gradPhi,dPhidt = Phi(0.0,x,do_gradient=True)
    dPhidx = torch.sum(gradPhi * dx, dim=1, keepdim=True)
    for k in range(10):
        h = 0.5 ** k
        Phit = Phi(h,x+h*dx,do_gradient=False)[0]

        E0 = torch.norm(Phic - Phit)
        E1 = torch.norm(Phic + h*(dPhidx +dPhidt) - Phit)

        print("h=%1.2e\tE0=%1.2e\tE1=%1.2e" % (h, E0, E1))

    print("Check derivatives w.r.t. weights")

    Phic, gradPhi, dPhidt = Phi(0.0, x, do_gradient=True)
    # w = torch.zeros(1, 4); w[0, -1] = 1.0  # test dPhidt w.r.t. theta
    # w = torch.zeros(1, 4); w[0, 0] = 1.0  # test Phi w.r.t. theta
    w = torch.zeros(1, 4);
    w[0, 1:3] = 1.0  # test gradPhi w.r.t. theta
    w = torch.ones(1, 4);

    print(w)
    F = torch.sum(w*torch.cat((Phic,gradPhi,dPhidt),dim=1))
    # F = torch.sum(dphidy)
    F.backward()
    W0 = net[0].K.data.clone()
    W = net[0].K
    dW = torch.randn_like(W0)
    dFdW = torch.sum(dW * W.grad)
    for k in range(20):
        h = 0.5 ** k
        Phi.net[0].K.data = W0 + h*dW
        Phit, gradPhit, dPhidtt = Phi(0.0, x, do_gradient=True)
        Ft = torch.sum(w * torch.cat((Phit, gradPhit, dPhidtt), dim=1))
        # Ft = torch.sum(dPhidyt)
        E0 = torch.norm(F - Ft)
        E1 = torch.norm(F + h * dFdW - Ft)

        print("h=%1.2e\tE0=%1.2e\tE1=%1.2e" % (h, E0, E1))




