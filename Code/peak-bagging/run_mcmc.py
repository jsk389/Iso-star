#!/usr/bin/python 

from __future__ import division
from __future__ import print_function

import emcee
import math
import shutil
import os
import model
from REGGAE.extract_params_xml import extract_local
import matplotlib.pyplot as pl
import numpy as np
import JSKemcee.priors as priors
import scipy.stats
from scipy.misc import factorial
from scipy.special import lpmv as legendre
import sys
import run_analysis
import gc


import Like


def JSK_Data(filename):
	f, p, = np.loadtxt(filename, usecols=(0, 1), unpack=True)
	bw = f[1]-f[0]
	return f, p, bw

class Model:
    def __init__(self, _f):
        self.nu = _f
        self.model = np.zeros(len(self.nu))

    def lorentzian(self, freq, width, amp, vis):
        f_minus = self.nu - freq
        bw = self.nu[1]-self.nu[0]
        T = 1.0 / (bw * 1e-6)
        line = 4.0 / width**2.0
        height = vis * 2.0 * (amp * 1e-6) **2.0 * T / \
                (np.pi * width * 1e-6 * T + 2)
        height *= 1e6
        return height /\
               (1.0 + line*(f_minus**2.0))

    def sphr_lm(self, l, theta):
        ell = int(l)
        amp = np.zeros(ell + 1)
        for mdx, m in enumerate(xrange(0, ell+1)):
            H = (factorial(ell - abs(m))/factorial(ell + abs(m))) \
                * legendre(m, ell, np.cos(np.radians(theta)))**2
            amp[mdx] = H
        return amp

    #def sphr_lm(self, l, theta):
    #    ell = int(l)
    #    amp = np.zeros(ell+1)
    #    amp[0] = np.cos(np.radians(theta))**2.0
    #    amp[1] = 0.5 * np.sin(np.radians(theta))**2.0
    #    return amp

    def full_model(self, back, freq, amp, width, split, inc):
        self.model[:] = back
        eea = self.sphr_lm(1, inc)
        self.model += self.lorentzian(freq, width, amp, eea[0])
        self.model += self.lorentzian(freq-split, width, amp, eea[1])
        self.model += self.lorentzian(freq+split, width, amp, eea[1])

        return self.model

    def __call__(self, params):
        return self.full_model(*params)

class Likelihood:
    def __init__(self, _f, _obs, _model):
        self.obs = _obs
        self.f = _f
        self.model = _model

    def __call__(self, params):
        mod = self.model(params)
        return -1.0 * np.sum(np.log(mod) + self.obs/mod)

class Prior:
    def __init__(self, _bounds, _gaussian):
        self.bounds = _bounds
        self.gaussian = _gaussian

    def __call__(self, p):
        # We'll just put reasonable uniform priors on all the parameters.
        if not all(b[0] < v < b[1] for v, b in zip(p, self.bounds)):
            return -np.inf
        lnprior = 0.0
        for idx, i in enumerate(self.gaussian):
            if i[1] != 0:
                lnprior += -0.5 * (p[idx] - i[0])**2 / i[1]**2
        # Place a prior on i - uniform in cos i
        lnprior += np.log(np.sin(np.radians(p[5])))
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
        return lp + like

def MCMC(params, like, prior, directory):
    ntemps, nwalkers, niter, ndims = 4, 1000, 1000, int(len(params))
    #sampler = emcee.EnsembleSampler(nwalkers, ndims, lnprob, \
    #                                                    threads=4)
    sampler = emcee.PTSampler(ntemps, nwalkers, ndims, like, \
                              prior, threads=4)
    param_names = ["Background", "Central Frequency", "Amplitude", \
									"Width", "Splitting", "Inclination"]

    p0 = np.zeros([ntemps, nwalkers, ndims])
    for i in range(ntemps):
        for j in range(nwalkers):
            p0[i,j,0] = params[0] + 1.0e-2*np.random.randn()
            p0[i,j,1] = params[1] + 1.0e-2*np.random.randn()
            p0[i,j,2] = params[2] + 1.0e-2*np.random.randn()
            p0[i,j,3] = params[3] + 1.0e-2*np.random.randn()
            p0[i,j,4] = params[4] + 1.0e-2*np.random.randn()
            p0[i,j,5] = np.degrees(np.arccos(np.random.rand(1)))
    #p0 = [params + 1.0e-4*np.random.randn(ndims) for k in range(nwalkers)]
    #p0 = np.zeros([nwalkers, ndims])
    #bounds = [(0.0, 2.0e2), (low_lims[a], upp_lims[a]),
    #                  (0.0, 50.0), (5.0e-4, 5.0), (0.0, 0.7),
    #                  (0.0, 90.0)]

    #for j in range(nwalkers):
    #    p0[j,0] = np.random.uniform(0.0, params[0]*10.0)
    #    p0[j,1] = 0.2*np.random.randn(1) + params[1]
    #    p0[j,2] = np.random.uniform(0.0, params[2]*10.0)
    #    p0[j,3] = np.random.uniform(5.0e-4, 5.0)
    #    p0[j,4] = np.random.uniform(0, 0.7)
    #    p0[j,5] = np.degrees(np.arccos(np.random.rand(1)))

    print('... burning in ...')
    for p, lnprob, lnlike in sampler.sample(p0, iterations=niter):
        pass
    #pos, _, _ = sampler.run_mcmc(p0, niter)
    sampler.reset()
    print('... running sampler ...')
    #sampler.run_mcmc(pos, niter)
    for p, lnprob, lnlike in sampler.sample(p, lnprob0=lnprob,
                                            lnlike0=lnlike,
                                            iterations=niter,
                                            thin=10):
        pass
    samples = sampler.chain[0,:, :, :].reshape((-1, ndims))
    #print('... checking for inc out of range')
	# Fold over inclination angle chains so i <= 90.0
    #for j in range(len(sampler.chain[:,0,5])):
    #    for k in range(len(sampler.chain[0,:,5])):
	#		if sampler.chain[j,k,5] > 90.0:
	#			sampler.chain[j,k,5] = 180.0 - \
	#						sampler.chain[j,k,5]
	#		if sampler.chain[j,k,5] < 0.0:
	#			sampler.chain[j,k,5] = 0.0 - \
	#						sampler.chain[j,k,5]

    print('... plotting chains')
    fig, axes = pl.subplots(6, 1, figsize=(8, 9))
    for l in range(len(params)):
	    axes[l].plot(samples[:,l].T, color="k", alpha=0.4)
	    axes[l].set_ylabel(parameters[l])
	    axes[l].set_xlabel("step number")
    fig.tight_layout(h_pad=0.0)
    # Check for previous file & remove if exists
    os.remove(directory+"/chain.pdf") if \
                os.path.exists(directory+"/chain.pdf") else None
    fig.savefig(directory+"/chain.png")
    pl.close()
    print('... saving samples')
    #np.savetxt(directory+'/samples.txt', samples)
    sampler.pool.terminate()

    return samples



#========================================================================#
# # # # # # # # # # # # # # # # MAIN # # # # # # # # # # # # # # # # # # #
#========================================================================#


if __name__=="__main__":

    # Read in list with KIC numbers of stars that are to be peak-bagged
    list_KIC = np.loadtxt(sys.argv[1], usecols=(0,), dtype=int)

    #main_dir = '/home/jsk389/Dropbox/Python/Peak-Bagging/Red_Giants/'
    #data_dir = '/home/jsk389/Dropbox/Python/Peak-Bagging/Red_Giants/Data/'
    #main_dir = '/home/jsk389/Dropbox/Python/Peak-Bagging/Red_Giants/Tom/knownseismiccases/'
    #data_dir = '/home/jsk389/Dropbox/Python/Peak-Bagging/Red_Giants/Tom/knownseismiccases/Data/'
    #main_dir = '/home/jsk389/Python/Angle_of_inc/'
    #data_dir = '/home/jsk389/Dropbox/Data/Kepler/RGs/'
    main_dir = '/home/jsk389/Dropbox/Python/Angle_of_inc/'
    data_dir = '/home/jsk389/Dropbox/Data/Kepler/Angle_of_inc/'

    for i in range(len(list_KIC)):

        print("Fitting KIC {0}".format(list_KIC[i]))


        # Import data in form of power spectrum
        #freq, power, bw = JSK_Data(main_dir+'/Data/PDCspec'+str(list_KIC[i])+'.pow')
        freq, power, bw = JSK_Data(data_dir+'kplr00'+str(list_KIC[i])+'_kasoc-psd_llc_v1.pow')

        print("Bin width = " + str(bw) + "uHz")

        #try:
        # Read in parameters file
        centre, amps, low_lims, upp_lims, backg, splittings, \
                inc = extract_local(str(main_dir)+str(list_KIC[i])+'/'+str(list_KIC[i])+'_params.xml')

        width = 0.02

        for a in range(len(centre)):

            # Define central freq of l=1 mode
            mu = centre[a]

            # Define freq fitting range
            power_sect = power[low_lims[a]/bw:upp_lims[a]/bw]
            freq_sect = freq[low_lims[a]/bw:upp_lims[a]/bw]


            # Set up prior bounds
            bounds = [(0.0, 2.0e2), (low_lims[a], upp_lims[a]),
                      (0.0, 50.0), (5.0e-4, 5.0), (0.0, 0.7),
                      (0.0, 90.0)]
            gaussian = [(0, 0), (mu, 0.2), (0, 0), (0, 0),
                        (0, 0), (0, 0)]

            # Define first guesses
            params = np.asarray([backg[a], centre[a], amps[a], \
                                    width, splittings[a], inc[a]]) 

            # Define prior, likelihood and model
            print('... setting up priors etc.')
            model = Model(freq_sect)
            like = Likelihood(freq_sect, power_sect, model)
            prior = Prior(bounds, gaussian)
            logprob = lnprob(prior, like)
            start = model(params)


            # Parameter names
            parameters = ["Background", "Central Frequency", \
                          "Amplitude",  "Width", "Splitting",\
                          "Inclination"]


            # Isotropic prior
            directory_first = main_dir+str(list_KIC[i])
            if not os.path.exists(directory_first):
                os.mkdir(directory_first)

            # Create folder to store data and plots in	
            # Isotropic prior
            directory = main_dir+str(list_KIC[i])+'/mode_'+str(a)+''
            if not os.path.exists(directory):
                os.mkdir(directory)

            print("Mode "+str(a+1)+" of "+str(len(centre))+"")

            samples = MCMC(params, like, prior, directory)
            gc.collect()
            print("... finished!")
        
            run_analysis.run_analysis(freq_sect, power_sect, 
                                          samples, directory)
        #except:
       #     print('Skipped')

