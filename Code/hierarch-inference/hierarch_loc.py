#!/usr/bin/python 

from __future__ import division
from __future__ import print_function

import emcee
import math
import shutil
import os
import matplotlib.pyplot as pl
import numpy as np
import scipy.stats
import sys
import corner
import gc
from scipy import integrate
from scipy.integrate import quad
from sini_inference_fns_mod import cosi_integrand, fisher_integrand, sine_fisher_integrand, cosine_integrand, cosine_fisher_integrand

# TO DO: Construct angle of inclination posteriors for each star
# TO DO: Work out priors on both kappa and mu (same as F&W, and uniform in mu?
# TO DO: Code up likelihood according to Hogg et al. 2010

def cosine_fisher_pdf(y, k, mu):

    if k == 0:
        norm = quad(cosine_fisher_integrand, 0, 1.0, args=(k, mu))[0]
        return np.ones(len(y)) / norm
    elif mu == 0:
        norm = quad(cosine_fisher_integrand, 0, 1.0, args=(k, mu))[0]
        return k / np.sinh(k) * np.exp(k*y) / norm
    else:
        norm = quad(cosine_fisher_integrand, 0, 1.0, args=(k, mu))[0]
        print(norm)
        return k / np.sinh(k) * np.exp(k*(y*np.cos(mu) + np.sqrt(1-y**2)*np.sin(mu))) / norm

def model(y, p):
    """
    Cosine of Fisher distribution with location
    """
    kappa, mu = p
    if kappa == 0.0:
        mod = np.ones(len(y))
    else:
        mod = kappa / np.sinh(kappa) * np.exp(kappa*(y*np.cos(mu) + np.sqrt(1-y**2)*np.sin(mu)))
    norm = quad(cosine_fisher_integrand, 0, 1.0, args=(kappa, mu))[0]
    return mod / norm

class Likelihood:
    def __init__(self, _samples, _priors):
        self.samples = _samples
        self.priors = _priors

    def uni_prior(self, sample):
        return 2.0 / (np.pi * np.sqrt(1 - sample)**2.0)

    def __call__(self, params):
        
        mod = np.zeros(np.shape(self.samples))
        for i in range(np.shape(self.samples)[1]):
            if self.priors == 'uni':
                mod[:,i] = model(self.samples[:,i], params) / \
                           self.uni_prior(self.samples[:,i])
            elif self.priors == 'iso':
                idx = np.argsort(self.samples[:,i])
                ss = self.samples[idx, i]                
                mod[:,i] = model(ss, params)
                #print(params, self.samples[:,i])
                #pl.plot(ss, mod[:,i])
                #pl.show()
        new_mod = mod.sum(axis=0) / float(np.shape(self.samples)[0])
        like = np.log(np.prod(new_mod))      
        if np.isfinite(like) == False:
            return -1.0e30
        return like        


class Prior:
    def __init__(self, _bounds):
        self.bounds = _bounds

    def __call__(self, p):
        # We'll just put reasonable uniform priors on all the parameters.
        if not all(b[0] < v < b[1] for v, b in zip(p, self.bounds)):
            return -np.inf
        lnprior = 0.0
        # Prior on kappa
        lnprior += np.log((1.0 + p[0]**2.0)**(-3.0/4.0))
        return lnprior

class lnprob:
    def __init__(self, _prior, _like):
        self.prior = _prior
        self.like = _like

    def __call__(self, p):
        lp = self.prior(p)
        if not np.isfinite(lp):
            return -np.inf
        like = self.like(p)
        #print(like)
        #print(lp, like, lp + like)
        return (lp + like)

def grid_hierarch(samps, like):
    """
    Basic hierarchical - grid
    """
    N = 100
    N_mu = 5
    kappa = np.linspace(0.01, 100.0, N)
    mu = np.linspace(0.01, np.pi/2.0 - 0.001, N_mu)
    power = np.zeros([N, N_mu])
    for i in range(len(kappa)):
        for j in range(len(mu)):
            print("{0} of {1}".format(i+1, len(kappa)))
            p = [kappa[i], mu[j]]
            power[i, j] = like(p)
    pl.figure(1)
    pl.contourf(kappa, mu, power.T, 50)
    #pl.plot(kappa, power)
    pl.xlabel(r'$\kappa$', fontsize=18)
    pl.ylabel(r'$\mu$', fontsize=18)
    pl.colorbar()
#    pl.ylabel(r'$\ln\left( L \right)$', fontsize=18)
    pl.figure(2)
    pl.plot(kappa, power.mean(axis=1))
    pl.figure(3)
    pl.plot(mu, power.mean(axis=0))
    pl.show()

def hpd(data, level) :
 	""" The Highest Posterior Density (credible) interval of data at 		"level" level.

	:param data: sequence of real values
	:param level: (0 < level < 1)
	"""

	d = list(data)
	d.sort()

	nData = len(data)
	nIn = int(round(level * nData))
	if nIn < 2 :
		raise RuntimeError("not enough data")

	i = 0
	r = d[i+nIn-1] - d[i]
	for k in range(len(d) - (nIn - 1)) :
		rk = d[k+nIn-1] - d[k]
		if rk < r :
			r = rk
			i = k

	assert 0 <= i <= i+nIn-1 < len(d)

	return (d[i], d[i+nIn-1])

def compute_values(x, level):
	"""
	Compute median, and credible interval using highest posterior density
	"""
	
	x_value = np.median(x)
	region = hpd(x, level)
	
	return [x_value, region[1]-x_value, x_value-region[0]]

def MCMC(params, like, lnprob, num):
    nwalkers, niter, ndims = 200, 500, int(len(params))
    sampler = emcee.EnsembleSampler(nwalkers, ndims, lnprob, \
                                                        threads=4)
    #param_names = [r'$\kappa$', r'$\mu$']

    p0 = [params + 1.0e-2*np.random.randn(ndims) for k in range(nwalkers)]

    print('... burning in ...')
    pos, _, _ = sampler.run_mcmc(p0, niter)
    sampler.reset()
    print('... running sampler ...')
    sampler.run_mcmc(pos, niter)

    for i in range(ndims):
        pl.figure()
        pl.hist(sampler.flatchain[:,i], 100, color="k", histtype="step")
        pl.title("Dimension {0:d}".format(i))
    pl.close()
    pl.show()

    samples = sampler.chain[:, :, :].reshape((-1, ndims))
    fig = corner.corner(samples, labels=[r'$\kappa$', r'$\mu$'])
    pl.show()


    level = 0.683
    kappa = compute_values(samples[:,0], level)
    mu = compute_values(samples[:,1], level)
    pl.hist(samples, bins=int(np.sqrt(len(samples))), normed=True, histtype='step')
    pl.axvline(kappa[0], linestyle='--', color='r')
    pl.axvline(kappa[0]+kappa[1], linestyle='--', color='r')
    pl.axvline(kappa[0]-kappa[2], linestyle='--', color='r')
    pl.xlabel(r'Concentration Parameter $\kappa$')
    pl.ylabel(r'Probability Density')
    pl.show()

    print("""
          Fit values:
          kappa = {0} + {1} - {2}
          mu = {3} + {4} - {5}
          """.format(kappa[0], kappa[1], kappa[2], mu[0], mu[1], mu[2]))
    #np.savetxt('samples_kappa_total.txt', samples)
    #np.savetxt(str(num)+'_limit_params.txt', np.c_[kappa, mu])

    #fig = corner.corner(samples, labels=param_names)
    #pl.savefig(str(num)+'_'+str(like.priors)+'_triangle.png')
    #pl.close()
    #pl.show()

    return kappa, mu, samples #samples[:,0], samples[:,1]

def run(fname, i):
    # MAKE SURE PROPERLY NORMALISED norm = np.sum(m[1:] * np.diff(np.sort(samps.flatten())))
    angles = np.linspace(0, 90, 10000)
    pl.plot(angles, model(angles, [0.0, 0.0])*np.pi/180.0, 'k')
    pl.show()

    posts = np.loadtxt(fname)
    m = len(posts) // 1000
    samps = posts[:m*1000].reshape((m,1000)).T
    # Convert into radians
    n, bins, _ = pl.hist(samps.flatten(), bins = np.sqrt(len(samps.flatten())), normed=True, histtype='step')
    
    #kap, mu = 0.0, 50.0
    #m = model(samps.flatten(), [kap, mu])# * np.pi/180.0
    #pl.plot(samps.flatten(), m, ',r')
    #pl.show()
    like = Likelihood(samps, 'iso')

    # kappa, mu
    params = [1.0, 0.2]
    bounds = [(-100.0, 100.0), (0.0, 90.0)]

    # Set up prior and likelihood
    prior = Prior(bounds)
    like = Likelihood(samps, 'iso')
    logprob = lnprob(prior, like)
    # Hierarchical on grid
    #grid_hierarch(samps, logprob)

    kappa = MCMC(params, like, logprob, i)
    #kappa_samps, mu_samps = 1, 1
    return kappa
#========================================================================#
# # # # # # # # # # # # # # # # MAIN # # # # # # # # # # # # # # # # # # #
#========================================================================#


if __name__=="__main__":

    posts = np.loadtxt(sys.argv[1])
    m = len(posts) // 1000
    samps = posts[:m*1000].reshape((m,1000)).T
#    samps = posts[:m*1000].reshape((1000, m))
    print(np.shape(samps))
    if np.max(samps.flatten()) > 1.0:       
        samps = np.cos(np.radians(samps))
    #samps = np.cos(np.radians(samps))
    # Convert into radians
    pl.hist(samps.flatten(), bins = int(np.sqrt(len(samps.flatten()))), normed=True, histtype='step')
    x = np.linspace(0, 1.0, 1000)
    #pl.plot(x, model(x, [3.69, 1.36]))
    pl.show()
    #samps = np.cos(np.radians(samps))
    """
    pl.plot(samps.flatten(), model(samps.flatten(), [0.0001]), 'r--')
    pl.ylabel(r'Probability Density', fontsize=18)
    pl.xlabel(r'$\cos i$', fontsize=18)
    pl.ylim(0,2)
    pl.show()
    # kappa, mu
    """
    params = [0.01, 0.01]
    bounds = [(0.0, 100.0), (0, np.pi/2.0)]

    # Set up prior and likelihood
    prior = Prior(bounds)
    like = Likelihood(samps, 'iso')
    logprob = lnprob(prior, like)
    # Hierarchical on grid
    #grid_hierarch(samps, like)

    kappa, mu, samples = MCMC(params, like, logprob, 0)
    pl.hist(samps.flatten(), bins = int(np.sqrt(len(samps.flatten()))), normed=True, histtype='step')
    inds = np.random.randint(0, len(samples), 100)
    for i in range(len(inds)):
        pl.plot(x, model(x, samples[inds[i],:]), alpha=0.1, color='r')
    pl.show()

    new_mod_p = np.zeros(len(x))
    new_mod_n = np.zeros(len(x))
    k11 = model(x, [kappa[0]+kappa[1], mu[0]+mu[1]])
    k12 = model(x, [kappa[0]+kappa[1], mu[0]-mu[2]])
    k21 = model(x, [kappa[0]-kappa[2], mu[0]+mu[1]])
    k22 = model(x, [kappa[0]-kappa[2], mu[0]-mu[2]])
    for i in range(len(new_mod_p)):
        new_mod_p[i] = np.max([k11[i], k12[i]])
        new_mod_n[i] = np.min([k21[i], k22[i]])

    pl.hist(samps.flatten(), bins = int(np.sqrt(len(samps.flatten()))), normed=True, histtype='step')
    pl.plot(x, model(x, [kappa[0], mu[0]]), 'r')
    pl.plot(x, new_mod_p, color='r', linestyle='--')
    pl.plot(x, new_mod_n, color='r', linestyle='--')
    pl.fill_between(x, new_mod_n, new_mod_p, facecolor='red', alpha=0.2)
    pl.show()
    
    
    
    sys.exit()
    kappa = [0.15151821876, 0.61048614, 0.57122651]
    print(kappa[0], kappa[1], float(kappa[1]))
    n, bins, _ = pl.hist(samps.flatten(), bins = int(np.sqrt(len(samps.flatten()))), normed=True, histtype='step')
    y = (bins[:-1] + bins[1:])/2
        
        
    kappa_range = np.random.normal(kappa[0], np.mean([kappa[1], kappa[2]]), 1000)
    avg_model = np.sum([model(y, i) for i in kappa_range], axis=0) / 1000.0
    print(len(avg_model))
    model1 = model(y, [kappa[0]+float(kappa[1])])
    model2 = model(y, [kappa[0]-float(kappa[2])])
    last_bin = np.where(y == y[model2 >= model1][-1])[0]
    print(last_bin)

    new_model1 = np.concatenate([model2[:last_bin], model1[last_bin:]])
    new_model2 = np.concatenate([model1[:last_bin], model2[last_bin:]])

    pl.plot(y, avg_model, 'k--')    
    pl.plot(y, model(y, [kappa[0]]), 'r--')
    #pl.plot(y, model2, 'r')
    #pl.plot(y, model1, 'r')
    pl.ylabel(r'Probability Density', fontsize=18)
    pl.xlabel(r'$\cos i$', fontsize=18)
    pl.ylim(0,2)
    pl.show()
