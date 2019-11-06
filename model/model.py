
import os, pickle

from . import standard_parameters as prm

import numpy as np
import scipy.stats as scts


def add_params(tr):

    tr.f_add_parameter('prm.pl_alpha', prm.pl_alpha)
    tr.f_add_parameter('prm.pl_xmin', prm.pl_xmin)

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

    np.random.seed(int(tr.v_idx))    
    print("Started process with id ", str(tr.v_idx))

    namespace = tr.prm.f_to_dict(short_names=True, fast_access=True)
    namespace['idx'] = tr.v_idx


    pid_pool = np.array(range(tr.Npool))
    
    lts = powerlaw_distribution(tr.Nprocess, tr.pl_alpha, xmin=tr.pl_xmin)
    lts_out = []

    init_pids = np.random.choice(pid_pool, replace=False,
                                 size=tr.Nprocess)

    # tracks [life time remaining (decreasing),
    #         counter (), 
    #         pool id (sampled without replacement),
    #         lifetime (fixed)]
    act_prcs = np.array([[lts, 0, pid, lts] for lts,pid in
                         zip(lts[:tr.Nprocess],init_pids)])

    for j in range(0, tr.Nsteps):

        ids = act_prcs[:,0]<0

        for prcs in act_prcs[ids]:
            lts_out.append([1, prcs[1], j, j-prcs[3], 1, -10, prcs[2]])

        # increase the counter if lifetime at end
        act_prcs[ids,1] += 1

        # choose new pool ids
        act_prcs[ids,2] = -10
        act_prcs[ids,2] = np.random.choice(np.setdiff1d(pid_pool,
                                                        act_prcs[:,2]),
                                           replace=False,
                                           size=np.sum(ids))

        # new lifetimes
        new_lts = powerlaw_distribution(np.sum(ids), tr.pl_alpha,
                                        xmin = tr.pl_xmin)
        act_prcs[ids,0], act_prcs[ids,3] = new_lts, new_lts
        
        # main step
        act_prcs[:,0] += -1
        

        
    # -1 for end of simulation "synapse didn't die"
    for prcs in act_prcs:
        lts_out.append([1, prcs[1], j+prcs[0], (j+prcs[0])-prcs[3], -1, -10, prcs[2]])

    raw_dir = './data/%.4d' %(tr.v_idx)
    
    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir)

    with open(raw_dir+'/namespace.p','wb') as pfile:
        pickle.dump(namespace,pfile)   
    with open(raw_dir+'/lts_in.p','wb') as pfile:
        pickle.dump(lts,pfile)   
    with open(raw_dir+'/lts.p','wb') as pfile:
        pickle.dump(lts_out,pfile)

        

        

        

    

    

    
