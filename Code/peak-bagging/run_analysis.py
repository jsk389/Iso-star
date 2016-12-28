#!/usr/bin/python 

from __future__ import division
from __future__ import print_function

import pickle
import pymultinest
import json
import emcee
import math
import shutil
import os
import FFT
import model
from REGGAE.extract_params_xml import extract_local
import matplotlib.pyplot as pl
import numpy as np
import scipy.stats
import sys

from plotutilities import *
from pyqt_fit import kde, kde_methods

import Like


def plot_chains(x, params, parameters, directory):
	"""
	Plot the chains from MultiNest
	"""
	
	fig, axes = pl.subplots(len(params),1, sharex=False, figsize=(20,18))
	fig.subplots_adjust(wspace=1.0, hspace=1.0)
	for i in range(0, len(params)):
		value = np.arange(0, len(x[:,i]))
		axes[i].plot(value, x[:,i], color='black')
		axes[i].set_ylabel(parameters[i])
		axes[i].set_xlabel(r'Chain')
	pl.savefig(directory+"/chains.pdf")
	pl.close()

def plot_marginals(x, params, parameters, directory):
	"""
	Plot marginals
	returns mode of inclination marginal
	"""

	# Select how much of chain to use in histograms
	#x = x[3.0*len(x)/4.0:, :]

	fig, axes = pl.subplots(2,3,figsize=\
							(20,10),facecolor='w',edgecolor='k')
	fig.subplots_adjust(wspace=0.5, hspace=0.5)
	axes = axes.ravel()

	mode = []

	for i in range(0, len(params)):
		# Compute HPD 68.3% credible interval
		cred_int = hpd(x[:,i], 0.683)
		ax = pl.subplot(2, 3, i+1)
		ax.set_rasterization_zorder(1)
		plotutils.plot_histogram_posterior(x[:,i], histtype='step',
											color='gray', normed=True, zorder=0)
		ax.axvline(x=cred_int[0], ymin=0, ymax=1e10, color='r', 
											linestyle='--', linewidth=2, zorder=0)
		ax.axvline(x=cred_int[1], ymin=0, ymax=1e10, color='r', 
											linestyle='--', linewidth=2, zorder=0)
		pl.axvspan(cred_int[0], cred_int[1], ymin=0, ymax=1e10, 
							hatch='/', color='red', alpha=0.1, zorder=0)
		if i ==5:
			est_large = kde.KDE1D(x[:,i], lower=0, upper=90,
								method=kde_methods.reflection)
			xxs, yys = est_large.grid()
			ax.plot(xxs, yys, 'b--', lw=2)
			ax.set_xlim(0.0, 90.0)
			ax.set_ylim(0.0, yys.max()+0.01)
			mode.append(xxs[yys==yys.max()][0])

		ax.set_xlabel(parameters[i])
		ax.set_ylabel(r'PPD')

	pl.tight_layout()
	pl.savefig(directory+"/marginals.eps", rasterized=True)
	pl.close()

	return mode

def plot_marginals2d(x, parameters, directory):
    import corner
    figure = corner.corner(x, labels = parameters)
    os.remove(directory+"/marginals_triangle.pdf") if \
        os.path.exists(directory+"/marginals_triangle.pdf") else None
    figure.savefig(directory+"/marginals_triangle.png")

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

def JSK_Data(filename):
	f, p, = np.loadtxt(filename, usecols=(0, 1), unpack=True)
	bw = f[1]-f[0]
	return f, p, bw

def run_analysis(freq_sect, power_sect, filename, directory):
    """
    Run analysis of emcee fit
    """

    parameters = ["Background", "Central Frequency", "Amplitude", \
						        "Width", "Splitting", "Inclination"]

    n_params = len(parameters)

    #x = np.loadtxt(filename)
    x = filename

    # Plot Marginals
    print("... plotting marginals")
    mode = plot_marginals(x, parameters, parameters, directory)

    print("... plotting 2d marginals")
    plot_marginals2d(x, parameters, directory)

    # Compute percentiles
    level = 0.683
    backg_mcmc = compute_values(x[:,0], level)
    centre_mcmc = compute_values(x[:,1], level)
    S_mcmc = compute_values(x[:,2], level)
    width_mcmc = compute_values(x[:,3], level)
    split_mcmc = compute_values(x[:,4], level)

    i_mcmc = [mode[-1], hpd(x[:,5], level)[1] - mode[-1], 										mode[-1]-hpd(x[:,5], level)[0]]

    #	backg_mcmc, centre_mcmc, S_mcmc, width_mcmc, split_mcmc, i_mcmc = list(backg_mcmc), list(centre_mcmc), list(S_mcmc), list(width_mcmc), list(split_mcmc), list(i_mcmc)

    print("""MCMC result:
    Background = {0[0]} +{0[1]} -{0[2]}
    Central Frequency = {1[0]} +{1[1]} -{1[2]}
    Amplitude = {2[0]} +{2[1]} -{2[2]}
    Width = {3[0]} +{3[1]} -{3[2]}
    Splitting = {4[0]} +{4[1]} -{4[2]}
    Inclination = {5[0]} +{5[1]} -{5[2]}
    """.format(backg_mcmc, centre_mcmc, S_mcmc, width_mcmc, split_mcmc, i_mcmc))

    new_params = [backg_mcmc[0], centre_mcmc[0], S_mcmc[0], \
				    width_mcmc[0], split_mcmc[0], i_mcmc[0]]
    new_params_plus = [backg_mcmc[1], centre_mcmc[1], S_mcmc[1], \
				    width_mcmc[1], split_mcmc[1], i_mcmc[1]]
    new_params_minus = [backg_mcmc[2], centre_mcmc[2], S_mcmc[2], \
				    width_mcmc[2], split_mcmc[2], i_mcmc[2]]

    # Plot "Best fit" model
    pl.figure() 
    pl.plot(freq_sect, power_sect, color='black', label='data')
    pl.plot(freq_sect, model.run(freq_sect, new_params, 1), '-', color='red', label='data')
    pl.xlim(freq_sect.min(), freq_sect.max())
    pl.savefig(directory+'mcmc_best_fit.pdf')
    pl.close('all')

    np.savetxt(directory+'/samples.txt', x[:,-1])

    np.savetxt(directory+'/limit_parameters.txt', \
		    np.c_[new_params, new_params_plus, new_params_minus])




#========================================================================
#								MAIN
#========================================================================

if __name__=="__main__":

	main_dir = '/home/jsk389/Python/Peak-Bagging/Red_Giants/Tom/knownseismiccases/'
	#main_dir = '/home/jsk389/Python/Peak-Bagging/Red_Giants/'

	# Import data in form of power spectrum
	freq, power, bw = JSK_Data(main_dir+'Data/PDCspec'+str(sys.argv[1])+'.pow')
	print("Bin width = " + str(bw) + "uHz")


	centre, amps, low_lims, upp_lims, backg, \
								splittings, inc = extract_local(main_dir+str(sys.argv[1])+'/'+str(sys.argv[1])+'_params.xml')

	for k in range(len(centre)):

		# Only supply frequency range over which fit was performed
		power_sect = power[low_lims[k]/bw:upp_lims[k]/bw]
		freq_sect = freq[low_lims[k]/bw:upp_lims[k]/bw]

		parameters = ["Background", "Central Frequency", "Amplitude", \
									"Width", "Splitting", "Inclination"]
	
		# Create folder to store data and plots in
	# Isotropic prior
	#	directory = main_dir+str(sys.argv[1])+'/mode_'+str(k)+''
	# Uniform prior
		directory = main_dir+str(sys.argv[1])+'/mode_'+str(k)+''
#		import threading, subprocess

#		if not os.path.exists("mode_"+str(k)+"/chains"): os.mkdir("mode_"+str(k)+"/chains")

		filename = directory+'/samples.txt'

		# Let us analyse the results
		run_analysis(freq_sect, power_sect, filename, directory)


