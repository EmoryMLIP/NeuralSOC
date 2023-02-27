import math
import torch
import os
import logging

def makedirs(dirname):
    """
    make the directory folder structure
    :param dirname: string path
    :return: void
    """
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    """structure for writing log file"""
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)
    with open(filepath, "r") as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger

def normpdf(x, mu, cov, a=1.0):
    """
    return pdf of the multivariate normal
    :param x:   tensor of shape nex-by-d
    :param mu:  tensor of shape 1-by-d
    :param cov: tensor of shape 1-by-d (the diagonal of the covariance matrix)
    :return: tensor of nex-by-1
    """
    nex,d = x.shape
    mu  = mu.view(1,d)
    cov = cov.view(1,d)

    denom = (2*math.pi)**(0.5*d) * torch.sqrt(torch.prod(cov))
    num   = a * torch.exp(-0.5 * torch.sum( (x - mu)**2 / cov , dim=1, keepdim=True))


    return num/denom