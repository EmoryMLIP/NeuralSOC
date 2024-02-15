# LQR Example
Linear Quadratic Regulator (LQR) problem represents a class of problems widely studied in deterministic and stochastic optimal control. We demonstrate our approach on a relatively simple example outlined below.
## Problem Description

We consider the following LQR problem:
$$
\begin{align}
\min_{\mathbf{u}_{t,\mathbf{x}}} \quad & \mathbb{E} \lbrace 50 \cdot ||\mathbf{z}(1)||^2 + \int_0^1 \frac{1}{2} ||\mathbf{u}(s) ||^2 {\rm d} s  \rbrace\\
\text{subject to  }   \quad & \text{d} {\mathbf{z}}(s) = \mathbf{u}(s) + \sigma \text{d} {W}(s), \quad s\in[0,1] \\
& \mathbf{z}(0) = \mathbf{x} = [-3,-3]^\top,
\end{align}
$$
where we set the constant σ = 0.4. This problem exhibits linear dynamics with quadratic cost, representing the standard form of a stochastic LQR problem.
## Solution Methods

1. **Riccati Equation:** 
   The most common approach to solving LQR problems is by solving the corresponding Riccati equation. We provide a MATLAB script (`LQR_ex.m`) to compute the optimal cost for the given LQR problem using the Riccati equation solver. Note that this requires the IVP Solver Toolbox, available at [IVP Solver Toolbox](https://www.mathworks.com/matlabcentral/fileexchange/103975-ivp-solver-toolbox).

2. **Finite Element Method (FEM) Solver:** 
   Another method involves solving the corresponding Hamilton-Jacobi-Bellman (HJB) equation using a FEM solver. We provide a Python script (`python 2DLQR_HJB_Fenics.py`) to approximate the solution using a finite element approximation. This script allows for a comparison between the Riccati solution and the FEM approximation.

3. **Neural Network Solver:**
   To test the neural network solver, run the following command:
	```
	python train_2d_LQR.py --prob Trajectory2 --net ResNet_hessquik --track_z True --n_iters 5000 --val_freq 100 --viz_freq 500 --print_freq 100 --lr 0.01 --beta '1.0, 10.0, 0.0, 10.0, 1.0, 0.0' --lr_freq 1600
	```
