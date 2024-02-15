% state matrix
A = [0   0;
     0   0];

% input matrix
B = [1 0;
     0 1];

% state weighting matrix
Q = [0.0   0;
     0   0.0];

% input weighting matrix
R = [1/2   0;
     0   1/2];

% terminal condition
PT = [50   0;
      0   50];

% solve riccati
[t,P] = solve_riccati_ode(A,B,Q,R,[],PT,[0,1]);

% deterministic LQR cost at [-3, 3]
z_Init = [-3,3];
aa = z_Init*P(:,:,1);
Vdlqr = sum(aa.*z_Init,2);

% stochastic constant
sigma = 0.16;
SOC_const = zeros(size(Vdlqr));
for i = 2:size(P,3)
    SOC_const=SOC_const+trace(sigma*(t(i)-t(i-1))*P(:,:,i));
end

% stochastic LQR cost
J_star = (Vdlqr+SOC_const);
save('Phi0_Riccati.dat',"J_star")