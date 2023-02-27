
"""
Some plotting
"""
import torch
import torch.nn as nn
import copy
from loss import control_obj, bsde_penalty, hjb_penalty, terminal_penalty
import os
from utils import count_parameters, makedirs, get_logger, normpdf

import pandas as pd
import numpy as np
import matplotlib
try:
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
except:
    matplotlib.use('Agg')  # for linux server with no tkinter
    import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import argparse

import math

parser = argparse.ArgumentParser('Plotter')
parser.add_argument('--log1'  , type=str, default='experiments/arr_pmp.npy') 
parser.add_argument('--log2'  , type=str, default='experiments/arr_randomwalk.npy') 
parser.add_argument('--pth1'  , type=str, default='experiments/BenchmarkProblem_Phi_OTflow_PMPSampler_track-z_False_betas_1_0_0_0_1_0_m64_2022_09_22_14_13_25_checkpt.pth') 
parser.add_argument('--pth2'  , type=str, default='experiments/BenchmarkProblem_Phi_OTflow_RandomWalkSampler_track-z_False_betas_1_0_0_0_0_0_m64_2022_09_22_16_22_57_checkpt.pth') 
parser.add_argument('--save'    , type=str, default='experiments/output/')
args = parser.parse_args()

device = torch.device('cpu')
makedirs(args.save)

from BenchmarkProblem import BenchmarkProblem
prob = BenchmarkProblem(type=2)
from Phi_OTflow import Phi_OTflow




if __name__ == '__main__':
    
    Phi = Phi_OTflow(4, 64, 100)
    from fsde import PMPSampler
    sampler = PMPSampler(Phi, prob, prob.t,prob.T,20)
    
    # PMP model
    checkpt = torch.load(args.pth1)
    Phi.net.load_state_dict(checkpt["state_dict"])
    Phi.w.load_state_dict(checkpt["w"])
    Phi.c.load_state_dict(checkpt["c"])
    Phi.A = checkpt["A"]
    
    x = prob.x_init(1)
    s, z, dw, Phiz, gradPhiz = sampler(x)

    clean_z_pmp = torch.zeros(21,2)
    for i in range(len(z)):
        clean_z_pmp[i,:] = z[i][0,[2,1]]
    clean_z_pmp = clean_z_pmp.detach().numpy()
    
    # RandomWalk model
    checkpt = torch.load(args.pth2)
    Phi.net.load_state_dict(checkpt["state_dict"])
    Phi.w.load_state_dict(checkpt["w"])
    Phi.c.load_state_dict(checkpt["c"])
    Phi.A = checkpt["A"]
    
    x = prob.x_init(1)
    s, z, dw, Phiz, gradPhiz = sampler(x)

    clean_z_rw = torch.zeros(21,2)
    for i in range(len(z)):
        clean_z_rw[i,:] = z[i][0,[2,1]]
    clean_z_rw = clean_z_rw.detach().numpy()
    
    plt.plot(clean_z_rw[:,0], clean_z_rw[:,1], 'o-', lw=3)
    plt.plot(clean_z_pmp[:,0], clean_z_pmp[:,1], 'o-', lw=3)
    plt.plot(3.0,3.0,'x', markersize=10)

    plt.xlim(-0.5, 3.5)
    plt.ylim(-0.5, 3.5)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.xlabel("dim 1")
    plt.ylabel("dim 2")
    plt.title('generated trajectories')

    plt.savefig(args.save + 'shifted_example.png', dpi=500, bbox_inches='tight')


    # plot the objective function log
    log_pmp = np.load(args.log1)
    log_rw  = np.load(args.log2)
    plt.figure()
    plt.plot(log_rw, lw=2)
    plt.plot(log_pmp, lw=2)
    plt.yscale('log')
    plt.xlabel('Number of Iteration')
    plt.ylabel('Objective function value')
    plt.legend(['RandomWalk','Feedback Control'])
    plt.savefig(args.save + 'shifted_example_log.png', dpi=500, bbox_inches='tight')
    
    
    











    