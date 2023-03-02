
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

from fenics import *
import math

parser = argparse.ArgumentParser('Plotter')
parser.add_argument('--resume'  , type=str, default='experiments/2dPTH') # NN models
parser.add_argument('--fem_sol'  , type=str, default='experiments/sol_array_compressed.npy') # fem solution
parser.add_argument('--save'    , type=str, default='experiments/output/')
args = parser.parse_args()

# logger
makedirs(args.save)
logger = get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))
logger.info(args)

device = torch.device('cpu')

from TrajectoryProblem import TrajectoryProblem2
prob = TrajectoryProblem2()
from Phi_hessQuik import Phi_hessQuik
import hessQuik.activations as act
import hessQuik.layers as lay
import hessQuik.networks as net



if __name__ == '__main__':
    torch.manual_seed(3)
    # Load fenics solution at selected time points
    nx, ny = 149,149
    mesh = RectangleMesh(Point(-3., -3.), Point(3., 3.), nx, ny) 
    parameters["reorder_dofs_serial"] = False 
    V = FunctionSpace(mesh, "Lagrange", 1)
    V_vec = VectorFunctionSpace(mesh, "Lagrange", 1)
    sol_array = np.load(args.fem_sol)
    u0 = Function(V)
    
    # load network model
    net = net.NN(
    lay.singleLayer(2 + 1, 32, act.tanhActivation()),
    lay.resnetLayer(32, 1.0, act.tanhActivation()),
    lay.singleLayer(32, 1, act.identityActivation())
    )
    Phi = Phi_hessQuik(net)
    from fsde import PMPSampler
    sampler = PMPSampler(Phi, prob, prob.t,prob.T,20)

    z_all = []
    n_models = len(os.listdir(args.resume))
    for filename in os.listdir(args.resume):
        logger.info(' ')
        logger.info("loading model: {:}".format(filename))
        logger.info(' ')
        ff = os.path.join(args.resume, filename)
        checkpt = torch.load(ff)
        Phi.net.load_state_dict(checkpt["state_dict"])
    
        # forward sampling
        # torch.manual_seed(3) # reproducibility
        x_init = torch.tensor([[-1.5,-1.5]]) + 0.4*torch.randn(256,2)
        s, z, _, Phiz, gradPhiz = sampler(x_init)

        z = torch.stack(z)
        z_all.append(z)
        
    
    z_all = torch.cat(z_all, dim=1).detach().numpy()
    np.save('z_all.npy',z_all)
    
    Phi_all = []
    sol_all = np.zeros((21,n_models*256,1))
    Phi0 = 0.0; J0 = 0.0
    sampler2 = PMPSampler(Phi, prob, prob.t,prob.T,200) # redefine sampler with smaller time steps for better accuracy
    for filename in os.listdir(args.resume):
        ff = os.path.join(args.resume, filename)
        checkpt = torch.load(ff)
        Phi.net.load_state_dict(checkpt["state_dict"])
        
        Phi_tmp = torch.zeros(21,n_models*256,1)
        # forward sampling
        for i in range(21):
            s = 0.05*i
            Phiz = Phi(s, torch.tensor(z_all[i,:,:]))
            Phi_tmp[i,:,:] = Phiz
            
        
        Phi_all.append(Phi_tmp)   
        
        # test and print J at x_init = [-1.5,-1.5]
        x = torch.tensor([-1.5, -1.5]).reshape(1,2)
        Phi0 = Phi0 + Phi(0.0,x).detach().numpy().item()
        x = x.repeat(2*4096,1)
        s, z, _, _, gradPhiz = sampler2(x)
        J, L, G = control_obj(Phi, prob, s, z, gradPhiz)
        J0 = J0 + J.detach().numpy()   
    
    for i in range(20):
        u0.vector()[:] = sol_array[-10*i-1,:,:].reshape(150**2)
        for j in range(n_models*256):
            sol_all[i,j,0] = u0(z_all[i,j,:])
    # hard code terminal condition
    u0.vector()[:] = 50*np.sum((V.tabulate_dof_coordinates()-np.array([[1.5,1.5]]))**2,1)
    for j in range(n_models*256):
            sol_all[-1,j,0] = u0(z_all[-1,j,:])
    
    
    Phi_all = torch.cat(Phi_all, dim=2).detach().numpy()
    np.save('Phi_all.npy',Phi_all)
    np.save('sol_all.npy', sol_all)

    
    
    
    # plotting
    fig, ax = plt.subplots()
    im = ax.scatter(z_all[0,:,0], z_all[0,:,1], c = np.mean(Phi_all[0,:,:],axis=1), vmin=np.min(sol_all[0,:,:]), vmax = np.max(sol_all[0,:,:]))
    ax.plot(1.5,1.5,'or')
    ax.set_xlim(-3,2)
    ax.set_ylim(-3,2)
    ax.set_aspect('equal')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax.set_xticks([], minor=False)
    ax.set_yticks([], minor=False)
    fig.savefig(args.save + '1_3.png', dpi=500, bbox_inches='tight')

    
    fig, ax = plt.subplots()
    im = ax.scatter(z_all[10,:,0], z_all[10,:,1], c = np.mean(Phi_all[10,:,:],axis=1), vmin=np.min(sol_all[10,:,:]), vmax = np.max(sol_all[10,:,:]))
    ax.plot(1.5,1.5,'or')
    ax.set_xlim(-3,2)
    ax.set_ylim(-3,2)
    ax.set_aspect('equal')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax.set_xticks([], minor=False)
    ax.set_yticks([], minor=False)
    fig.savefig(args.save + '2_3.png', dpi=500, bbox_inches='tight')

    
    fig, ax = plt.subplots()
    im = ax.scatter(z_all[18,:,0], z_all[18,:,1], c = np.mean(Phi_all[18,:,:],axis=1), vmin=np.min(sol_all[18,:,:]), vmax = np.max(sol_all[18,:,:]))
    ax.plot(1.5,1.5,'or')
    ax.set_xlim(-3,2)
    ax.set_ylim(-3,2)
    ax.set_aspect('equal')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax.set_xticks([], minor=False)
    ax.set_yticks([], minor=False)
    fig.savefig(args.save + '3_3.png', dpi=500, bbox_inches='tight')


    fig, ax = plt.subplots()
    im = ax.scatter(z_all[0,:,0], z_all[0,:,1], c = sol_all[0,:,:])
    ax.plot(1.5,1.5,'or')
    ax.set_xlim(-3,2)
    ax.set_ylim(-3,2)
    ax.set_aspect('equal')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax.set_xticks([], minor=False)
    ax.set_yticks([], minor=False)
    fig.savefig(args.save + '1_4.png', dpi=500, bbox_inches='tight')
    

    fig, ax = plt.subplots()
    im = ax.scatter(z_all[10,:,0], z_all[10,:,1], c = sol_all[10,:,:])
    ax.plot(1.5,1.5,'or')
    ax.set_xlim(-3,2)
    ax.set_ylim(-3,2)
    ax.set_aspect('equal')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax.set_xticks([], minor=False)
    ax.set_yticks([], minor=False)
    fig.savefig(args.save + '2_4.png', dpi=500, bbox_inches='tight')
    

    fig, ax = plt.subplots()
    im = ax.scatter(z_all[18,:,0], z_all[18,:,1], c = sol_all[18,:,:])
    ax.plot(1.5,1.5,'or')
    ax.set_xlim(-3,2)
    ax.set_ylim(-3,2)
    ax.set_aspect('equal')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax.set_xticks([], minor=False)
    ax.set_yticks([], minor=False)
    fig.savefig(args.save + '3_4.png', dpi=500, bbox_inches='tight')
    
    
    fig, ax = plt.subplots()
    im = ax.scatter(z_all[0,:,0], z_all[0,:,1], c = np.abs(sol_all[0,:,:]  - np.mean(Phi_all[0,:,:],axis=1,keepdims=True)))
    ax.set_xlim(-3,2)
    ax.set_ylim(-3,2)
    ax.set_aspect('equal')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax.set_xticks([], minor=False)
    ax.set_yticks([], minor=False)
    fig.savefig(args.save + '1_5.png', dpi=500, bbox_inches='tight')
    
    
    fig, ax = plt.subplots()
    im = ax.scatter(z_all[10,:,0], z_all[10,:,1], c = np.abs(sol_all[10,:,:] - np.mean(Phi_all[10,:,:],axis=1, keepdims=True)))
    ax.set_xlim(-3,2)
    ax.set_ylim(-3,2)
    ax.set_aspect('equal')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax.set_xticks([], minor=False)
    ax.set_yticks([], minor=False)
    fig.savefig(args.save + '2_5.png', dpi=500, bbox_inches='tight')
    
    
    fig, ax = plt.subplots()
    im = ax.scatter(z_all[18,:,0], z_all[18,:,1], c = np.abs(sol_all[18,:,:] - np.mean(Phi_all[18,:,:],axis=1, keepdims=True)))
    ax.set_xlim(-3,2)
    ax.set_ylim(-3,2)
    ax.set_aspect('equal')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax.set_xticks([], minor=False)
    ax.set_yticks([], minor=False)
    fig.savefig(args.save + '3_5.png', dpi=500, bbox_inches='tight')
    
    
    # single model result
    fig, ax = plt.subplots()
    im = ax.scatter(z_all[0,:,0], z_all[0,:,1], c = Phi_all[0,:,-1], vmin=np.min(sol_all[0,:,:]), vmax = np.max(sol_all[0,:,:]))
    ax.plot(1.5,1.5,'or')
    ax.set_xlim(-3,2)
    ax.set_ylim(-3,2)
    ax.set_aspect('equal')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax.set_xticks([], minor=False)
    ax.set_yticks([], minor=False)
    fig.savefig(args.save + '1_1.png', dpi=500, bbox_inches='tight')

    
    fig, ax = plt.subplots()
    im = ax.scatter(z_all[10,:,0], z_all[10,:,1], c = Phi_all[10,:,-1], vmin=np.min(sol_all[10,:,:]), vmax = np.max(sol_all[10,:,:]))
    ax.plot(1.5,1.5,'or')
    ax.set_xlim(-3,2)
    ax.set_ylim(-3,2)
    ax.set_aspect('equal')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax.set_xticks([], minor=False)
    ax.set_yticks([], minor=False)
    fig.savefig(args.save + '2_1.png', dpi=500, bbox_inches='tight')


    fig, ax = plt.subplots()
    im = ax.scatter(z_all[18,:,0], z_all[18,:,1], c = Phi_all[18,:,-1], vmin=np.min(sol_all[18,:,:]), vmax = np.max(sol_all[18,:,:]))
    ax.plot(1.5,1.5,'or')
    ax.set_xlim(-3,2)
    ax.set_ylim(-3,2)
    ax.set_aspect('equal')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax.set_xticks([], minor=False)
    ax.set_yticks([], minor=False)
    fig.savefig(args.save + '3_1.png', dpi=500, bbox_inches='tight')
    
    
    fig, ax = plt.subplots()
    im = ax.scatter(z_all[0,:,0], z_all[0,:,1], c = Phi_all[0,:,-2], vmin=np.min(sol_all[0,:,:]), vmax = np.max(sol_all[0,:,:]))
    ax.plot(1.5,1.5,'or')
    ax.set_xlim(-3,2)
    ax.set_ylim(-3,2)
    ax.set_aspect('equal')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax.set_xticks([], minor=False)
    ax.set_yticks([], minor=False)
    fig.savefig(args.save + '1_2.png', dpi=500, bbox_inches='tight')

    
    fig, ax = plt.subplots()
    im = ax.scatter(z_all[10,:,0], z_all[10,:,1], c = Phi_all[10,:,-2], vmin=np.min(sol_all[10,:,:]), vmax = np.max(sol_all[10,:,:]))
    ax.plot(1.5,1.5,'or')
    ax.set_xlim(-3,2)
    ax.set_ylim(-3,2)
    ax.set_aspect('equal')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax.set_xticks([], minor=False)
    ax.set_yticks([], minor=False)
    fig.savefig(args.save + '2_2.png', dpi=500, bbox_inches='tight')


    fig, ax = plt.subplots()
    im = ax.scatter(z_all[18,:,0], z_all[18,:,1], c = Phi_all[18,:,-2], vmin=np.min(sol_all[18,:,:]), vmax = np.max(sol_all[18,:,:]))
    ax.plot(1.5,1.5,'or')
    ax.set_xlim(-3,2)
    ax.set_ylim(-3,2)
    ax.set_aspect('equal')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax.set_xticks([], minor=False)
    ax.set_yticks([], minor=False)
    fig.savefig(args.save + '3_2.png', dpi=500, bbox_inches='tight')



    # test and print J at x_init = [-1.5,-1.5]
    logger.info('fixing initial state x = [-1.5,-1.5]')
    logger.info('Phi = {:.4e}'.format(Phi0/n_models))
    logger.info('J = {:.4e}'.format(J0/n_models))
    
    # same for fenics solution
    u0.vector()[:] = sol_array[-1,:,:].reshape(150**2)
    x = np.array([[-1.5,-1.5]])
    logger.info('Fenics Phi = {:.4e}'.format(u0(x.flatten())))
    # Fenics J 
    dt = 0.005
    z = np.zeros([201,4096,2])
    x = np.array([[-1.5,-1.5]])
    x = x.repeat(4096,0)
    z[0,:,:] = x
    
    grad_phi = np.zeros_like(z)
    sig = np.array([0.2,-0.4,-0.4,0.2]).reshape(2,2)

    for i in range(200):
        u = Function(V)
        u.vector()[:] = sol_array[-i-1,:,:].reshape(150**2)
        h = project(grad(u),V_vec)

        grad_x = np.zeros([4096,2])
        for j in range(4096):
            grad_x[j,:] = h(x[j,:])
        grad_phi[i,:,:] = grad_x
        dw = np.random.randn(4096,2)*np.sqrt(dt)
        x = x - grad_x * dt + dw@sig
        z[i+1,:,:] = x
    
    L = 0.0
    for i in range(200):
        s = dt*i
        L = L + dt * torch.mean(prob.L(s,torch.tensor(z[i,:,:]),torch.tensor(grad_phi[i,:,:])))

    G = torch.mean(prob.g(torch.tensor(z[-1,:,:]))[0])
    
    
    logger.info('Fenics J = {:.4e}'.format((L+G).detach().numpy()))
    
    
    absolute_err = np.zeros((n_models,3))
    relative_err = np.zeros((n_models,3))
    for i in range(n_models):
        absolute_err[i,0] = np.mean(np.abs(Phi_all[0,:,i] - sol_all[0,:,0]))
        absolute_err[i,1] = np.mean(np.abs(Phi_all[10,:,i] - sol_all[10,:,0]))
        absolute_err[i,2] = np.mean(np.abs(Phi_all[18,:,i] - sol_all[18,:,0]))

        relative_err[i,0] = np.mean(np.abs(Phi_all[0,:,i] - sol_all[0,:,0]) / sol_all[0,:,0])
        relative_err[i,1] = np.mean(np.abs(Phi_all[10,:,i] - sol_all[10,:,0]) / sol_all[10,:,0])
        relative_err[i,2] = np.mean(np.abs(Phi_all[18,:,i] - sol_all[18,:,0]) / sol_all[18,:,0])
        
    
    logger.info('Mean absolute error, s = 0:  {:.4e}'.format(np.mean(absolute_err[:,0])))
    logger.info('Mean absolute error, s = 0.5:  {:.4e}'.format(np.mean(absolute_err[:,1])))
    logger.info('Mean absolute error, s = 0.9:  {:.4e}'.format(np.mean(absolute_err[:,2])))
    
    logger.info('Mean relative error, s = 0:  {:.4e}'.format(np.mean(relative_err[:,0])))
    logger.info('Mean relative error, s = 0.5:  {:.4e}'.format(np.mean(relative_err[:,1])))
    logger.info('Mean relative error, s = 0.9:  {:.4e}'.format(np.mean(relative_err[:,2])))
 
    
    logger.info('STD absolute error, s = 0:  {:.4e}'.format(np.std(absolute_err[:,0])))
    logger.info('STD absolute error, s = 0.5:  {:.4e}'.format(np.std(absolute_err[:,1])))
    logger.info('STD absolute error, s = 0.9:  {:.4e}'.format(np.std(absolute_err[:,2])))
    
    logger.info('STD relative error, s = 0:  {:.4e}'.format(np.std(relative_err[:,0])))
    logger.info('STD relative error, s = 0.5:  {:.4e}'.format(np.std(relative_err[:,1])))
    logger.info('STD relative error, s = 0.9:  {:.4e}'.format(np.std(relative_err[:,2])))



    # additional plotting
    fig, ax = plt.subplots(1, 3, figsize=(15,5))
    # 1
    img = ax[0].imshow(np.flipud(sol_array[-1,25:124,25:124]),extent=(-2,2,-2,2), )
    ax[0].plot(1.5,1.5,'or')
    # plt.colorbar()
    ax[0].set_title("value function $\Phi_{FEM}$, s=0.0")
    
    # 2
    N = 48
    x = torch.linspace(-2.0,2.0,steps=N)
    y = torch.linspace(-2.0,2.0,steps=N)
    X,Y = torch.meshgrid(x,y)
    XY = torch.cat((X.reshape(-1,1), Y.reshape(-1,1)),1)
    XY = XY.detach().numpy()
    nx, ny = 149,149
    mesh = RectangleMesh(Point(-3., -3.), Point(3., 3.), nx, ny) # 
    parameters["reorder_dofs_serial"] = False 
    V = FunctionSpace(mesh, "Lagrange", 1)
    V_vec = VectorFunctionSpace(mesh, "Lagrange", 1)
    u0 = Function(V)
    u0.vector()[:] = sol_array[-1,:,:].reshape(150**2)
    h = project(grad(u0),V_vec)
    grad_phi = np.zeros_like(XY)
    for i in range(XY.shape[0]):
        grad_phi[i,:] = h(XY[i,:])

    ax[1].quiver(XY[::11, 0], XY[::11, 1], -grad_phi[::11, 0], -grad_phi[::11, 1])  # arrow at every 11th pt
    ax[1].set_title(r'$-\nabla_z \Phi_{FEM} (z, s=0)$')
    ax[1].set_aspect('equal')

    # 3
    dt = 0.05
    n_samples = 16
    z = np.zeros([21,n_samples,2])
    x = np.array([[-1.5,-1.5]])
    x = x.repeat(n_samples,0) + 0.4 * np.random.randn(n_samples,2)
    z[0,:,:] = x

    grad_phi = np.zeros_like(z)
    sig = np.array([0.2,-0.4,-0.4,0.2]).reshape(2,2)

    for i in range(20):
        u = Function(V)
        u.vector()[:] = sol_array[-10*i-1,:,:].reshape(150**2)
        h = project(grad(u),V_vec)

        grad_x = np.zeros([n_samples,2])
        for j in range(n_samples):
            grad_x[j,:] = h(x[j,:])
        grad_phi[i,:,:] = grad_x
        dw = np.random.randn(n_samples,2)*np.sqrt(dt)
        x = x - grad_x * dt + dw@sig
        z[i+1,:,:] = x


    x = torch.linspace(-2.0,2.0,steps=96)
    y = torch.linspace(-2.0,2.0,steps=96)
    X,Y = torch.meshgrid(x,y)
    XY = torch.cat((X.reshape(-1,1), Y.reshape(-1,1)),1)
    mu = torch.zeros(2); cov = 0.4*torch.ones(2) 
    V_map = 50. * normpdf(XY, mu=mu.view(1, -1), cov=cov.view(1, -1))
    V_map = V_map.view(X.shape[0],X.shape[1])

    ax[2].pcolormesh(x,y,V_map,cmap='hot')
    for i in range(n_samples):
        ax[2].plot(z[:,i,0], z[:,i,1],'-o')
        ax[2].plot(1.5,1.5,'X')
        ax[2].set_xlim(-2., 2.)
        ax[2].set_ylim(-2., 2.)
        ax[2].set_aspect('equal', adjustable='box')
        ax[2].set_title('Trajectory examples')


    axins1 = inset_axes(
        ax[0],
        width="5%",  # width: 50% of parent_bbox width
        height="90%",  # height: 5%
        loc="right",
    )
    # fig.colorbar(img, ax=[ax[0]], shrink=0.8, location='right')
    fig.colorbar(img, cax=axins1)

    fig.savefig(args.save + '2d_phi_compile.png', dpi=500, bbox_inches='tight')
