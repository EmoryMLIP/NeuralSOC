import torch
import torch.autograd as autograd

class AbstractOCProblem:
    def __init__(self):
        return self

    def calH(self,s,z,p,u,M=None):
        """

        calH = f(s,z,u)'*p - L(s,z,u) + trace(sigma^2 M)

        :param s: time
        :param z: state
        :param p: adjoint state
        :param u: control
        :param M: adjoint variable in stochastic OC problems
        :return: calH
        """
        H = torch.sum(p*self.f(s,z,u),dim=1,keepdim=True)  - self.L(s,z,u)
        if M is not None:
            H = H + self.sigma2_Lap(s,z,M)
        return H

    def test_Hamiltonian(self,s,z,p,M=None):
        """
        Test 1: | calH(s,z,p,u_star) - Hamiltonian(s,z,p) |

        Test 2: derivative check for gradpH computed in Hamiltonian

        :param s:
        :param z:
        :param p:
        :param M:
        :return:
        """

        H_true = self.calH(s, z, p, self.u_star(s, z, p))
        H,gradpH = self.Hamiltonian(s,z,p)
        err_H = torch.norm(H_true - H)
        print("err_H = %1.2e" % (err_H))

        dp = torch.randn_like(p)
        dd = torch.sum(dp*gradpH,dim=1,keepdim=True)
        for k in range(20):
            h = (0.5)**(k)
            Ht = self.Hamiltonian(s,z,p+h*dp)[0]

            E0 = torch.norm(H-Ht)
            E1 = torch.norm(H + h*dd -Ht)
            print("h=%1.2e\tE0=%1.2e\tE1=%1.2e" %(h,E0,E1))


    def test_g(self,z):
        g,gradg = self.g(z)
        dz = torch.randn_like(z)
        dgdz = torch.sum(gradg*dz,dim=1,keepdim=True)
        for k in range(20):
            h = (0.5)**(k)
            gt = self.g(z+h*dz)[0]

            E0 = torch.norm(g-gt)
            E1 = torch.norm(g + h*dgdz -gt)
            print("h=%1.2e\tE0=%1.2e\tE1=%1.2e" %(h,E0,E1))

    def test_u_star(self,s,z,p):
        """
        test the feedback form. Note that

        u_star = argmax_u calH(s,z,p,u)

        and therefore |\nabla_u calH(s,z,p,u_star)| \approx 0

        :param s:
        :param z:
        :param p:
        :return:
        """
        ut = torch.tensor(self.u_star(s,z,p),requires_grad=True)
        H = self.calH(s,z,p,ut)
        dH = autograd.grad(H, ut, torch.ones([ut.shape[0], 1], device=ut.device, dtype=ut.dtype), retain_graph=True,
                      create_graph=True)[0]
        err = torch.norm(dH)
        # du = torch.randn_like(ut)
        # dd = torch.sum(du*dH,dim=1,keepdim=True)
        # for k in range(20):
        #     h = (0.5)**(k)
        #     Ht = self.calH(s, z, p, ut + h*du)
        #
        #     E0 = torch.norm(H-Ht)
        #     E1 = torch.norm(H + h*dd -Ht)
        #     print("h=%1.2e\tE0=%1.2e\tE1=%1.2e" %(h,E0,E1))
        return err
