import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.nn.functional import pad


class PhiNN(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def __str__(self):
        return self.net.__str__()

    def forward(self,s,x,do_gradient=False):
        y = pad(x,[0,1,0,0],value=s)
        if do_gradient is False:
            return self.net(y)
        else: # use AD to compute derivatives
            yt = y.clone()
            if yt.requires_grad == False:
                yt.requires_grad = True
            Phic = self.net(yt)
            dPhiy = autograd.grad(Phic,yt,torch.ones([x.shape[0], 1],device=yt.device,dtype=yt.dtype),retain_graph=True, create_graph= True)[0]
            dPhidx = dPhiy[:,:-1]
            dPhidt = dPhiy[:,-1]
            return Phic, dPhidx, dPhidt.unsqueeze(1)


if  __name__ == '__main__':

    d = 2
    nex = 2
    width = [8]

    net = nn.Sequential(
        nn.Linear(d+1, width[0]),
        nn.Tanh(),
        nn.Linear(width[-1], 1))
    Phi = PhiNN(net)
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
    W0 = net[0].weight.data.clone()
    W = net[0].weight
    dW = torch.randn_like(W0)
    dFdW = torch.sum(dW * W.grad)
    for k in range(20):
        h = 0.5 ** k
        Phi.net[0].weight.data = W0 + h*dW
        Phit, gradPhit, dPhidtt = Phi(0.0, x, do_gradient=True)
        Ft = torch.sum(w * torch.cat((Phit, gradPhit, dPhidtt), dim=1))
        # Ft = torch.sum(dPhidyt)
        E0 = torch.norm(F - Ft)
        E1 = torch.norm(F + h * dFdW - Ft)

        print("h=%1.2e\tE0=%1.2e\tE1=%1.2e" % (h, E0, E1))




