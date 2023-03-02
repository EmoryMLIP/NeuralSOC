# NeuralHJB
Code for method proposed in [A Neural Network Approach for Stochastic Optimal Control](https://arxiv.org/pdf/2209.13104.pdf).
## Setup
Run these commands.
```
pip install -r install.txt 
```
A portion of the experiments require the installation of FeniCS, installation guide for FeniCS can be found [here](https://fenicsproject.org/download/archive/). FeniCS can also be installed on Colab, refer to [this](https://fem-on-colab.github.io/packages.html) guide to run the experiments on Colab.
 
Note: For FEninCS based experiments, Anaconda installation of FEniCS version 2019.1.0 on Linux system with Ubuntu 20.04.3 is used with the following dependencies:
- FEniCS==2019.1.0
- matplotlib==3.4.3
- numpy==1.21.2

## Experiments
#### For section 4.1
2D obstacle avoiding problem
```
python NeuralSOC/train_2d.py --prob Trajectory2 --net ResNet_hessquik --track_z True --n_iters 6000 --val_freq 100 --viz_freq 400 --print_freq 100 --lr 0.01 --beta '1.0, 1.0, 0.0, 1.0, 1.0, 0.0' --lr_freq 1800
```
Solving the HJB equation corresponding to the 2D trajectory planning problem using a finite element method
```
python 2D_HJB_Fenics.py
```
Plotter for section 4.1
```
python plotter.py
```
#### For section 4.2
For a more detailed comparison regarding the 100D Benchmark problem we recommend our separate implementation using Tensorflow, which can be found [here](https://github.com/EmoryMLIP/FBSNNs), that said a pytorch version of the problem is also available.
```
python train_100d_new.py --prob Benchmark --net ResNet_OTflow --track_z True --n_iters 50000 --val_freq 100 --viz_freq 100 --print_freq 100 --lr_freq 20000 --lr 0.001 --beta '1.0, 1.0, 1.0, 0.0, 20.0, 0.0' --m 64
```
100D problem with shifted target state
```
python NeuralSOC/train_100d.py --prob Benchmark2 --net ResNet_OTflow --track_z True --n_iters 20000 --val_freq 100 --viz_freq 100 --print_freq 100 --lr_freq 8000 --lr 0.002 --beta '10.0, 0.1, 0.1, 0.0, 1.0, 0.0' --m 64
```
Plotter for section 4.2
```
python plotter2.py
```


## References
This code unifies the approaches in the following papers:

- [A machine learning framework for solving high-dimensional mean field game and mean field control problems](https://www.pnas.org/doi/10.1073/pnas.1922204117)
- [A Neural Network Approach for High-Dimensional Optimal Control Applied to Multiagent Path Finding](https://ieeexplore.ieee.org/document/9786046)

It also leverages the efficient package for computing neural networks and their derivatives:
- [`hessQuik`: Fast Hessian computation of composite functions](https://joss.theoj.org/papers/10.21105/joss.04171)

## Acknowledgements
This material is in part based upon work supported by the Department of Energy RISE ASCR 20-023231  and the US AFOSR Grant FA9550-20-1-0372. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the funding agencies.
