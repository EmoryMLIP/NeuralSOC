import torch
import torch.nn as nn
import time
from loss import control_obj, bsde_penalty, hjb_penalty, terminal_penalty
from QuadcopterProblem import QuadcopterProblem
from Phi_OTflow import Phi_OTflow
import os
from utils import count_parameters, makedirs, get_logger
import datetime

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser('Quadcopter Problem')
parser.add_argument('--sampler', choices=['PMPSampler'],   type=str, default='PMPSampler')
parser.add_argument("--nt_train"    , type=int, default=100, help="number of time steps for training")
parser.add_argument("--nt_val", type=int, default=200, help="number of time steps for validation")
parser.add_argument("--n_train"    , type=int, default=256, help="number of training examples")
parser.add_argument('--beta'  , type=str, default='0.1, 0.1, 0.1, 0.1, 1.0, 0.0') # BSDE penalty, Terminal, grad terminal, HJB, J, Phi(0)-J; Note: Terminal already has a weight of 100
parser.add_argument('--save'    , type=str, default='experiments/oc/run', help="define the save directory")
parser.add_argument('--gpu'     , type=int, default=0, help="send to specific gpu")
parser.add_argument('--prec'    , type=str, default='single', choices=['single','double'], help="single or double precision")
parser.add_argument("--n_val"    , type=int, default=4096, help="number of validation examples")
parser.add_argument("--n_plot"    , type=int, default=10, help="number of plot examples")
parser.add_argument('--n_iters', type=int, default=6000)
parser.add_argument('--optim' , type=str, default='adam', choices=['adam'])
parser.add_argument('--lr'    , type=float, default=0.01)
parser.add_argument('--lr_freq' , type=int  , default=1600, help="how often to decrease lr")
parser.add_argument('--lr_decay', type=float, default=0.5, help="how much to decrease lr")
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--net', choices=['ResNet_OTflow'],   type=str, default='ResNet_OTflow')
parser.add_argument('--m'     , type=int, default=128, help="NN width")
parser.add_argument('--track_z' , type=str, choices=['False', 'True'], default='False', help="to track gradients for state")
parser.add_argument('--resume'  , type=str, default=None, help="for loading a pretrained model")
parser.add_argument('--val_freq', type=int, default=50, help="how often to run model on validation set")
parser.add_argument('--print_freq', type=int, default=50, help="how often to print results to log")
parser.add_argument('--viz_freq', type=int, default=100, help="how often to plot visuals") # must be >= val_freq
parser.add_argument('--sample_freq',type=int, default=1, help="how often to resample training data")

args = parser.parse_args()

beta = [float(item) for item in args.beta.split(',')]
sStartTime = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

# logger
logger = {}
makedirs(args.save)
logger = get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__), saving = True)
logger.info("start time: " + sStartTime)
logger.info(args)

if __name__ == '__main__':
    
    if args.resume is not None:
        logger.info(' ')
        logger.info("loading model: {:}".format(args.resume))
        logger.info(' ')
        
        checkpt = torch.load(args.resume, map_location=lambda storage, loc: storage)
        args.m = checkpt['args'].m
        args.nTh = checkpt['args'].nTh
        
    # set precision and device
    if args.prec == 'double':
        argPrec = torch.float64
    else:
        argPrec = torch.float32
    torch.set_default_dtype(argPrec)
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

    prob = QuadcopterProblem()
    Phi = Phi_OTflow(2, args.m, prob.d)
    
    if args.resume is not None:
        Phi.net.load_state_dict(checkpt["state_dict"])
        Phi.net = Phi.net.to(argPrec).to(device)
        Phi.w.load_state_dict(checkpt["w"]); Phi.w = Phi.w.to(argPrec).to(device)
        Phi.c.load_state_dict(checkpt["c"]); Phi.c = Phi.c.to(argPrec).to(device)
        Phi.A = checkpt["A"]; Phi.A = Phi.A.to(argPrec).to(device)
    
    from fsde import PMPSampler
    sampler = PMPSampler(Phi, prob, prob.t,prob.T,args.nt_train)
    sampler_val = PMPSampler(Phi,prob,prob.t,prob.T,args.nt_val)
    
    lr = args.lr
    optim = torch.optim.Adam(Phi.parameters(), lr=lr, weight_decay=args.weight_decay)
    strTitle = prob.__class__.__name__ + '_' + Phi._get_name() + '_' + sampler._get_name() + '_track-z_' + args.track_z + '_betas_{:}_{:}_{:}_{:}_{:}_{:}_m{:}_'.format(
                        int(beta[0]), int(beta[1]), int(beta[2]),int(beta[3]), int(beta[4]), int(beta[5]), args.m) + sStartTime  # add a flag before start time for tracking
    logger.info("---------------------- Network ----------------------------")
    logger.info(Phi.net)
    logger.info("----------------------- Problem ---------------------------")
    logger.info(prob)
    logger.info("------------------------ Sampler (train) --------------------------")
    logger.info(sampler)
    logger.info("------------------------ Sampler (validation) --------------------------")
    logger.info(sampler)
    logger.info("--------------------------------------------------")
    logger.info("beta={:}".format(args.beta))
    logger.info("Number of trainable parameters: {}".format(count_parameters(Phi.net)))
    logger.info("--------------------------------------------------")
    logger.info(str(optim))  # optimizer info
    logger.info("dtype={:} device={:}".format(argPrec, device))
    logger.info("n_train={:} n_val={:} n_plot={:}".format(args.n_train, args.n_val, args.n_plot))
    logger.info("maxIters={:} val_freq={:} viz_freq={:}".format(args.n_iters, args.val_freq, args.viz_freq))
    logger.info("saveLocation = {:}".format(args.save))
    logger.info(strTitle)
    logger.info("--------------------------------------------------\n")
    
    
    columns = ["step","loss","Phi0","L","G","cHJB","cBSDE","cPhi","cBSDEfin","cBSDEgrad","lr"]
    logger.info(columns)
    train_hist = pd.DataFrame(columns=columns)
    val_hist = pd.DataFrame(columns=columns)
    
    xp = prob.x_init(10)
    s,z,dw,Phiz,gradPhiz = sampler(xp)
    fig = plt.figure()
    ax = plt.gca()
    figPath = args.save + '/figures/'
    if not os.path.exists(os.path.dirname(figPath)):
        os.makedirs(os.path.dirname(figPath))
    prob.render(s,z,dw,Phi,os.path.join(figPath, '%s_iter_%s.png' % (strTitle, 'pre-training')))
    
    best_loss = float('inf')
    bestParams = None
    Phi.net.train()
    
    xv = prob.x_init(args.n_val)
    xp = prob.x_init(args.n_plot)
    
    makedirs(args.save)
    start_time = time.time()
    for itr in range(args.n_iters - 1):
        if itr==0 or itr % args.sample_freq == 0:
            x = prob.x_init(args.n_train)
            
        optim.zero_grad()
        s, z, dw, Phiz, gradPhiz = sampler(x)
        if args.track_z == 'False':
            optim.zero_grad()
            # (Phiz, gradPhiz) = (None,None)
        J, L, G = control_obj(Phi, prob, s, z, gradPhiz)
        term_Phi, term_gradPhi = terminal_penalty(Phi, prob, s, z, dw, Phiz, gradPhiz)
        cBSDE = bsde_penalty(Phi, prob, s, z, dw, Phiz, gradPhiz)
        cHJB = hjb_penalty(Phi, prob, s, z, dw, Phiz, gradPhiz)
        Phi0 = Phi(s[0],x)
        cPhi = torch.mean(torch.abs(Phi0-J))
        loss = beta[0]*cBSDE + beta[1]*term_Phi + beta[2]*term_gradPhi + beta[3] * cHJB + beta[4]*J + beta[5] * cPhi
        loss.backward()
        optim.step()
        
        train_hist.loc[len(train_hist.index)] = [itr,loss.item(),torch.mean(Phi0).item(),L.item(),G.item(),cHJB.item(),cBSDE.item(),cPhi.item(),term_Phi.item(),term_gradPhi.item(),lr]

        # printing
        if itr % args.print_freq == 0:
            ch = train_hist.iloc[-1:]
            if itr >0:
                ch.columns=11*['']
                ch.index.name=None
                log_message = (ch.to_string().split("\n"))[1]
            else:
                log_message = ch
            logger.info(log_message)
            
        if itr % args.val_freq == 0 or itr == args.n_iters:
            s, z, dw, Phiz, gradPhiz = sampler_val(xv)
            J, L, G = control_obj(Phi, prob, s, z, gradPhiz)
            term_Phi, term_gradPhi = terminal_penalty(Phi, prob, s, z, dw, Phiz, gradPhiz)
            cBSDE = bsde_penalty(Phi, prob, s, z, dw, Phiz, gradPhiz)
            cHJB = hjb_penalty(Phi, prob, s, z, dw, Phiz, gradPhiz)
            Phi0 = Phi(s[0], xv)
            cPhi = torch.mean(torch.abs(Phi0 - J))
            test_loss = beta[0] * cBSDE + beta[1] * term_Phi + beta[2] * term_gradPhi + beta[3] * cHJB + beta[4] * J + beta[5] * cPhi
            val_hist.loc[len(val_hist.index)] = [itr,test_loss.item(), torch.mean(Phi0).item(), L.item(),
                                                                 G.item(), cHJB.item(), cBSDE.item(), cPhi.item(),
                                                                 term_Phi.item(), term_gradPhi.item(),lr]
            if test_loss.item() < best_loss:
                best_loss = test_loss.item()
                makedirs(args.save)
                bestParams = Phi.net.state_dict()
                torch.save({
                    'args': args,
                    'A': Phi.A,
                    'w': Phi.w.state_dict(),
                    'c': Phi.c.state_dict(),
                    'state_dict': bestParams,
                    }, os.path.join(args.save, strTitle + '_checkpt.pth'))
                        
                print('save new best model')

        # shrink step size
        if (itr + 1) % args.lr_freq == 0:
            lr *= args.lr_decay
            for p in optim.param_groups:
                p['lr'] *= lr

        if itr % args.viz_freq == 0:
            s, z, dw, Phiz, gradPhiz = sampler_val(xp)
            fig = plt.figure(figsize=plt.figaspect(1.0))
            fig.suptitle('iteration=%d' % (itr))
            prob.render(s, z, dw, Phi,  os.path.join(figPath, '%s_iter_%d.png' % (strTitle, itr)))


    elapsed = time.time() - start_time
    print('Training time: %.2f secs' % (elapsed))
    train_hist.to_csv(os.path.join(args.save, '%s_train_hist.csv' % (strTitle )))
    val_hist.to_csv(os.path.join(args.save, '%s_val_hist.csv' % (strTitle )))