
import os, pickle

from . import standard_parameters as prm

import numpy as np
import scipy.stats as scts


def add_params(tr):

    tr.f_add_parameter('prm.pl_alpha', prm.pl_alpha)

    tr.f_add_parameter('prm.Nprocess', prm.Nprocess)
    tr.f_add_parameter('prm.Npool', prm.Npool)
    tr.f_add_parameter('prm.Nsteps', prm.Nsteps)


def powerlaw_distribution(samples, alpha, xmin=1):

    # using the transformation method (Press 2007, pp. 362)
    # to get random variates for variable with probability
    # density
    #
    #       f(t) = c * t^(-a)    for x >= x_min     (1)
    #
    # where c = (a-1) * x_min^(a-1). For this first
    # calculate F(t) the cumulative distribution function
    #
    #           F(t) = 1 - (t/x_min)^(-a+1)         (2)
    #
    # Then get then inverse
    #
    #       F^(-1)(z) = x_min * (1-z)^(1/(-a+1))    (3)
    #
    # Finally, if r are variates of a uniformly distributed
    # random variable in [0,1), then by the transformation
    # method
    #
    #      y = F^(-1)(r) = = x_min * (1-r)^(1/(-a+1)) 
    #
    # are variates of a RV distributed according to (1).

    assert (xmin > 0)

    rs = np.random.uniform(size=samples)
    ys = xmin*(1-rs)**(1/(-1*alpha+1))

    return ys
    


        
def run_model(tr):

    pass


    

    

    
