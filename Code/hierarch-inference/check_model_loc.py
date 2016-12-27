#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import sys

from scipy import interpolate, integrate
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

    # Plot in inc
    for i in range(len(kappa)):
        for j in range(len(mu)):
            fish[:,i] = inv_sampl(angles, model(angles, [kappa[i], mu[j]]), N)
            plt.plot(angles, model(angles, [kappa[i], mu[j]]), color=color[i])
            if mu[j] == np.pi/2.0:
                plt.hist(fish[:,i], bins=np.sqrt(N), normed=True, \
                        histtype='step', color=color[i], linewidth=2, \
                        label=r'$\kappa$ = '+str(kappa[i])+'; $\mu$ = $\pi$/2')
            else:
                plt.hist(fish[:,i], bins=np.sqrt(N), normed=True, \
                        histtype='step', color=color[i], linewidth=2, \
                        label=r'$\kappa$ = '+str(kappa[i])+'; $\mu$ = '+str(mu[j]))
    plt.xlim(0, np.pi/2.0)
    plt.ylabel(r'Probability Density', fontsize=18)
    plt.xlabel(r'$i_{\mathrm{s}}$ (radians)', fontsize=18)
    plt.legend(loc='best')
    plt.savefig('fisher_loc_inc.png')
    plt.show()

    for i in range(len(kappa)):
        for j in range(len(mu)):
            fish[:,i] = inv_sampl(angles, model(angles, [kappa[i], mu[j]]), N)
            cos_fish[:,i] = np.cos(fish[:,i])
            plt.plot(np.cos(angles), cosine_fisher_pdf(np.cos(angles), [kappa[i], mu[j]]), color=color[i])
            if mu[j] == np.pi/2.0:
                plt.hist(cos_fish[:,i], bins=np.sqrt(N), normed=True, \
                        histtype='step', color=color[i], linewidth=2, \
                        label=r'$\kappa$ = '+str(kappa[i])+'; $\mu$ = $\pi$/2')
            else:
                plt.hist(cos_fish[:,i], bins=np.sqrt(N), normed=True, \
                        histtype='step', color=color[i], linewidth=2, \
                        label=r'$\kappa$ = '+str(kappa[i])+'; $\mu$ = '+str(mu[j]))
    plt.xlim(0, 1.0)
    plt.ylabel(r'Probability Density', fontsize=18)
    plt.xlabel(r'$\cos i_{\mathrm{s}}$', fontsize=18)
    plt.legend(loc='best')
    plt.savefig('fisher_loc_cosine_inc.png')
    plt.show()
