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
from scipy import interpolate
from scipy import integrate
from scipy.integrate import quad
from sini_inference_fns_mod import cosi_integrand, fisher_integrand, sine_fisher_integrand, cosine_integrand, cosine_fisher_integrand

# TO DO: Construct angle of inclination posteriors for each star
# TO DO: Work out priors on both kappa and mu (same as F&W, and uniform in mu?
# TO DO: Code up likelihood according to Hogg et al. 2010

def cosine_fisher_pdf(y, p):

    k, mu = p
    if k == 0:
        norm = quad(cosine_fisher_integrand, 0, 1.0, args=(k, mu))[0]
        print(norm)
        return np.ones(len(y)) / norm
    elif mu == 0:
        norm = quad(cosine_fisher_integrand, 0, 1.0, args=(k, mu))[0]
        return k / np.sinh(k) * np.exp(k*y) / norm
    else:
        norm = quad(cosine_fisher_integrand, 0, 1.0, args=(k, mu))[0]
        print("NORM: ", norm)
        return k / np.sinh(k) * np.exp(k*(y*np.cos(mu) + np.sqrt(1-y**2)*np.sin(mu))) / norm

def cosine_model(y, p):
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

def model(theta, p):
    """
    Fisher distribution with location
    """
    kappa, mu = p
    if kappa == 0.0:
        mod = np.sin(theta)       
        norm = quad(fisher_integrand, 0, np.pi/2.0, args=(kappa, mu))[0]
        return mod / norm
    else:
        mod = kappa / np.sinh(kappa) * np.exp(kappa*np.cos(theta - mu)) * np.sin(theta)
        norm = quad(fisher_integrand, 0, np.pi/2.0, args=(kappa, mu))[0]
    return mod / norm


def inv_sampl(vals, pdf, N):
	cdf = pdf.cumsum()
	cdf /= cdf.max()
	inv_cdf = interpolate.interp1d(cdf, vals)

	r = np.random.uniform(cdf.min()+0.0001, 1.0, N)

	return inv_cdf(r)

#========================================================================#
# # # # # # # # # # # # # # # # MAIN # # # # # # # # # # # # # # # # # # #
#========================================================================#


if __name__=="__main__":

    N = 1e4
    angles = np.linspace(0, np.pi/2.0, N)
    kappa = [0.0, 0.5, 1.0, 2.0, 5.0]
    mu = [np.pi/2.0]
    fish = np.zeros([N, len(kappa)])
    cos_fish = np.zeros([N, len(kappa)])
    color=['cyan', 'blue', 'green', 'red', 'magenta', 'orange', 'saddlebrown', 
           'olive', 'gold']
    for i in range(len(kappa)):
        for j in range(len(mu)):
  
            fish[:,i] = inv_sampl(angles, model(angles, [kappa[i], mu[j]]), N)
            pl.plot(angles, model(angles, [kappa[i], mu[j]]))
            pl.hist(fish[:,i], bins=np.sqrt(N), normed=True, \
                    histtype='step', color=color[i], linewidth=2, \
                    label=r'$\kappa$ = '+str(kappa[i])+'; $\mu$ = '+str(mu[j]))           
            pl.show()
            cos_fish[:,i] = np.cos(fish[:,i])
            pl.plot(np.cos(angles), cosine_fisher_pdf(np.cos(angles), [kappa[i], mu[j]]))
            pl.hist(cos_fish[:,i], bins=np.sqrt(N), normed=True, \
                    histtype='step', color=color[i], linewidth=2, \
                    label=r'$\kappa$ = '+str(kappa[i])+'; $\mu$ = '+str(mu[j]))
            #y = np.cos(np.radians(angles))
            #pl.plot(y, cos_fisher(y, kappa[i]), linestyle='-', \
            #        linewidth=2, color=color[i], label=r'$\kappa$ = '+str(kappa[i]))
            pl.show()
    pl.ylabel(r'Probability Density', fontsize=18)
    pl.xlabel(r'$\cos i$', fontsize=18)
    pl.legend(loc='best')
    pl.show()
 
    #kappa = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0]    
    for i in range(len(kappa)):
        fish[:,i] = inv_sampl(angles, fisher(angles, kappa[i]), N)
        #pl.hist(fish[:,i], bins=np.sqrt(N), normed=True, \
        #        histtype='step', color=color[i], linewidth=2, \
        #        label=r'$\kappa$ = '+str(kappa[i]))
        y = angles
        pl.plot(y, fisher(y, kappa[i]), linestyle='-', \
                linewidth=2, color=color[i], label=r'$\kappa$ = '+str(kappa[i]))
    pl.ylabel(r'Probability Density', fontsize=18)
    pl.xlabel(r'$i$', fontsize=18)
    pl.ylim([0,0.03])
    pl.legend(loc='best')
    pl.show()


