# An LQR Example
LQR problems represent a large class of problems in both deterministic and stochastic optimal control problems. We here test our approach on a relatively simple example. We consider the following problem 

$$
\begin{aligned}
\min_{\mathbf{u}_{t,\mathbf{x}}} \quad & \mathbb{E} \left\{ 50 \cdot \|\mathbf{z}(1)\|^2 + \int_0^1 \frac{1}{2} \|\mathbf{u}(s) \|^2 {\rm d} s  \right\}\\
\text{subject to  }   \quad & \text{d} {\mathbf{z}}(s) = \mathbf{u}(s) + \sigma \text{d} {W}(s), \;\; s\in[0,1] \\
& \mathbf{z}(0) = \mathbf{x} = [-3,-3]^\top,
\end{aligned}
$$
here we select constant $\sigma =0.4$. The problem has linear dynamics with quadratic cost, and therefore exhibits the standard form of a stochastic LQR problem. 

The most common way to treat a LQR problem is to solve the corresponding Riccati equation, with many tutorials available, one can test using the example with the following code:
```
Update Matlab code
```
One way to verify the Riccati solution is to solve the HJB equation using a FEM solver
```
python 2DLQR_HJB_Fenics.py
```
One can test our neural network solver through
```
python train_2d_LQR.py --prob Trajectory2 --net ResNet_hessquik --track_z True --n_iters 5000 --val_freq 100 --viz_freq 500 --print_freq 100 --lr 0.01 --beta '1.0, 10.0, 0.0, 10.0, 1.0, 0.0' --lr_freq 1600

```
