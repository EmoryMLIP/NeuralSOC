import torch
import torch.nn as nn
from torch.nn.functional import pad
import time
from loss import control_obj, bsde_penalty, hjb_penalty, terminal_penalty
from QuadcopterProblem import QuadcopterProblem
from fsde import PMPSampler
from Phi_OTflow import Phi_OTflow
import os
from utils import count_parameters, makedirs, get_logger
import datetime

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser('Quadcopter evaluation')
parser.add_argument('--pth1'  , type=str, default='experiments/singlequad_nn_checkpt.pth') 
parser.add_argument('--pth2'  , type=str, default='experiments/QuadcopterProblem_Phi_OTflow_PMPSampler_track-z_False_betas_0_0_0_0_1_0_m128_2023_09_17_15_06_19_checkpt.pth') 
parser.add_argument('--save'    , type=str, default='experiments/output/')
args = parser.parse_args()

args = parser.parse_args()

# logger
logger = {}
makedirs(args.save)
logger = get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))
logger.info(args)

device = torch.device('cpu')

if __name__ == '__main__':
    
    prob = QuadcopterProblem()
    Phi1 = Phi_OTflow(2, 128, prob.d)
    logger.info(' ')
    logger.info("loading model 1: {:}".format(args.pth1))
    logger.info(' ')
    
    checkpt = torch.load(args.pth1, map_location=lambda storage, loc: storage)
    Phi1 = Phi_OTflow(2, 128, 12)
    Phi1.A.data = checkpt["state_dict"]["A"]
    Phi1.w.weight.data = checkpt["state_dict"]["w.weight"]
    Phi1.c.weight.data = checkpt["state_dict"]["c.weight"]
    Phi1.c.bias.data = checkpt["state_dict"]["c.bias"]

    Phi1.net.layers[0].weight.data = checkpt["state_dict"]["N.layers.0.weight"]
    Phi1.net.layers[0].bias.data = checkpt["state_dict"]["N.layers.0.bias"]
    Phi1.net.layers[1].weight.data = checkpt["state_dict"]["N.layers.1.weight"]
    Phi1.net.layers[1].bias.data = checkpt["state_dict"]["N.layers.1.bias"]
    
    logger.info(' ')
    logger.info("loading model 2: {:}".format(args.pth2))
    logger.info(' ')
    
    checkpt = torch.load(args.pth2, map_location=lambda storage, loc: storage)
    Phi2 = Phi_OTflow(2, 128, 12)
    Phi2.net.load_state_dict(checkpt["state_dict"])
    Phi2.w.load_state_dict(checkpt["w"])
    Phi2.c.load_state_dict(checkpt["c"])
    Phi2.A = checkpt["A"]
    
    J0_ODE = 0.0
    J0_SDE = 0.0
    sampler1 = PMPSampler(Phi1, prob, prob.t,prob.T,200) 
    sampler2 = PMPSampler(Phi2, prob, prob.t,prob.T,200) 
    
    x = torch.tensor([-1.5, -1.5,-1.5,0,0,0,0,0,0,0,0,0]).reshape(1,12)
    x = x.repeat(3000,1)
    for i in range(5):
        # total of 15k sample trajectories
        s, z, _, _, gradPhiz = sampler1(x)
        J, L, G = control_obj(Phi1, prob, s, z, gradPhiz)
        J0_ODE = J0_ODE + J.detach().numpy()   
    
        s, z, _, _, gradPhiz = sampler2(x)
        J, L, G = control_obj(Phi2, prob, s, z, gradPhiz)
        J0_SDE = J0_SDE + J.detach().numpy()   
    
    
    logger.info('J deterministic= {:.4e}'.format(J0_ODE/5))
    logger.info('J stochastic= {:.4e}'.format(J0_SDE/5))
    
    # make a figure
    x0 = torch.Tensor([[-1.5,-1.5,-1.5]]) + 0.3*torch.randn(5,3)
    x0 = pad(x0, [0,prob.d-3,0,0], value=0)
    sampler = PMPSampler(Phi2, prob, prob.t,prob.T,100) 
    s, z, dw, Phi, gradPhiz = sampler(x0)
    prob.render(s,z,dw,Phi,'final_plot_quad.png')
    
