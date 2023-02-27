import torch
import torch.nn as nn
from Phi import PhiNN

def control_obj(Phi,prob,s,z,gradPhiz):
    L = 0.0;
    for i in range(s.shape[0]-1):
        ds = s[i+1]-s[i]
        if gradPhiz is None:
            Phizi, gradPhizi, _ = Phi(s[i], z[i], do_gradient=True)
        else:
            gradPhizi = gradPhiz[i]

        ui = prob.u_star(s[i],z[i],-gradPhizi)
        L = L + ds * torch.mean(prob.L(s[i],z[i],ui))

    G = torch.mean(prob.g(z[-1])[0])

    return L + G, L, G

def terminal_penalty(Phi,prob,s,z,dw,Phiz,gradPhiz):
    if Phiz is None or gradPhiz is None:
        Phizi, gradPhizi, _ = Phi(s[-1], z[-1], do_gradient=True)
    else:
        (Phizi, gradPhizi) = (Phiz[-1],gradPhiz[-1])

    G, gradG = prob.g(z[-1])
    cHJBfin = torch.mean(torch.abs(Phizi - G.view(-1, 1)))
    cHJBgrad = torch.mean(torch.sum(torch.abs(gradPhizi - gradG), dim=1))
    return cHJBfin, cHJBgrad

# switch between L1 and L2 norm for error accumulation if necessary
def hjb_penalty(Phi,prob,s,z,dw,Phiz,gradPhiz):
    cHJB=0.0
    for i in range(s.shape[0]-1):
        ds = s[i + 1] - s[i]
        if prob.__class__.__name__ == 'TrajectoryProblem2':
            Phizi, gradPhizi, dtPhizi, hessPhizi = Phi(s[i + 1], z[i + 1], do_gradient=True, do_Hessian=True)  
            res = - dtPhizi - 0.5* prob.tr_sigma2_M(s[i + 1], z[i + 1],hessPhizi) + prob.Hamiltonian(s[i],z[i],-gradPhizi,None)[0]
        else:
            Phizi, gradPhizi, dtPhizi, LapPhi = Phi(s[i + 1], z[i + 1], do_gradient=True, do_Laplacian=True)
            res = - dtPhizi - 0.5* prob.sigma(s[i + 1], z[i + 1])**2*LapPhi + prob.Hamiltonian(s[i],z[i],-gradPhizi,None)[0]


        cHJB = cHJB + torch.mean(torch.abs(res),dim=0) *ds

    return cHJB

def bsde_penalty(Phi,prob,s,z, dw,Phiz,gradPhiz):
    cBSDE = 0.0;
    if gradPhiz is None:
        Phizi, gradPhizi, _ = Phi(s[0], z[0], do_gradient=True)
    else:
        Phizi = Phiz[0]
        gradPhizi = gradPhiz[0]

    for i in range(s.shape[0]-1):
        ds = s[i + 1] - s[i]
        if gradPhiz is None:
            Phizi1, gradPhizi1, _ = Phi(s[i+1], z[i+1], do_gradient=True)
        else:
            Phizi1 = Phiz[i+1]
            gradPhizi1 = gradPhiz[i+1]

        ui = prob.u_star(s[i], z[i], -gradPhizi)
        dL = torch.mean(prob.L(s[i], z[i], ui))

        BSDEerr = Phizi1 - Phizi + dL * ds - torch.sum(gradPhizi * prob.sigma_mv(s[i],z[i],dw[i]), dim=1, keepdim=True)   # check signs
        cBSDE = cBSDE +  ds*torch.mean(torch.sum(torch.abs(BSDEerr), dim=1))

        (Phizi,gradPhizi) = (Phizi1, gradPhizi1)

    return cBSDE



if  __name__ == '__main__':
    from TrajectoryProblem import TrajectoryProblem
    from fsde import EulerMaryama
    d = 2
    nex = 2
    width = [8]


    net = nn.Sequential(
        nn.Linear(d + 1, width[0]),
        nn.Tanh(),
        nn.Linear(width[-1], 1))

    Phi = PhiNN(net)

    prob = TrajectoryProblem()
    nex = 300
    nt = 100
    beta = (1.0, 1.0, 1.0)
    prob.sigma=0.0 # deterministic
    x = prob.x_init(nex)

    integrator = EulerMaryama(0.0,1.0, nt)

    loss = FBSDEloss(integrator,beta)

    Jc, L, G, cBSDE, cBSDEfin, cBSDEgrad = loss(Phi,prob,x)
    G.backward()

    W0 = net[0].weight.data.clone()
    W = net[0].weight
    dW = torch.randn_like(W0)
    dFdW = torch.sum(dW * W.grad)
    for k in range(20):
        h = 0.5 ** k
        Phi.net[0].weight.data = W0 + h * dW
        Jt, Lt, Gt, cBSDEt, cBSDEfint, cBSDEgradt = loss(Phi,prob,x)
        # Ft = torch.sum(dPhidyt)
        E0 = torch.norm(G - Gt)
        E1 = torch.norm(G + h * dFdW - Gt)

        print("h=%1.2e\tE0=%1.2e\tE1=%1.2e" % (h, E0, E1))
