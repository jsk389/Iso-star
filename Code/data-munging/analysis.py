#!/usr/bin/python

from __future__ import division

import glob
import matplotlib.pyplot as pl
import numpy as np
import os
import scipy.stats
import sys

from plotutilities import *
from useful_functions import flatten

from helper_fns import, hpd, compute_values

def calc_heights(a, gamma):
    """
    Calculate mode heights - eqn from ... Fletcher (...)
    """
    T = 365.25 * 86400.0 * 4.0
    height = 2.0 * a **2.0 * T / (np.pi * gamma * T + 2)
    return height / 1e6

def extract_data(root_dir, list_dirs, cut='sims', output_type='stars'):
    """
    Main function to extract the data in given directories.
    The arguments can be used to dictate what parameters are to be used if
    any cuts are needed or anything different needs to be returned.

    ----------------
    Parameters:

    cut : string: dictates cuts to be made to data.
                - "sims" makes cut which was determined from fitting of artificial
                  data -> width greater than bin-width and SNR (i.e. H/B in our case)
                  of greater than 18.9 to ensure reliable fits.
                - "kepmag" makes cut determined to be at median of Kepler magnitude
                - "redsplit" makes cut determined to be at median of reduced splitting
                  of sample (width / splitting)

    output_type: string: dictates which results to return.
                        - "stars" will compute results for stellar inclination angles
                          as expected.
                        - "modes" will compute results for individual oscillation modes
    """
    # Create lists for output
    if output_type == "stars":
        all_samples = []
        n_modes = []
        average_inc = []
        average_error_pos = []
        average_error_neg = []
    elif output_type == "modes":
        sigma_mode = []
        sigma_posts = []
        inc_modes = []
        inc_tot = []

    # Loop over directories for each star
    for i in range(len(list_dirs)):
        # Firstly identify the samples for each star
        print("Reading in star {0} of {1}".format(i+1, len(list_dirs)))
        dirs = glob.glob(rootdir+str(list_dirs[i])+'/*/samples.txt')
        param_dirs = [s.strip('samples.txt') for s in dirs]

        save_dir = rootdir#+list_dirs[i]

        samples_tot = []
        mds = 0
        # Loop over directories for each star
        for j in range(len(dirs)):
            backg, frequency, amp, width, split, inc = np.loadtxt(param_dirs[j]+'limit_parameters.txt', usecols=(0,), unpack=True)
            errs = np.loadtxt(param_dirs[j]+'limit_parameters.txt', usecols=(1,2), unpack=True)
            #errs = np.mean(errs.T, axis=1)
            #if inc > 10.0:
            #frac_err.append(errs[0]/backg)
            #inc_tot.append(backg)
            snr = calc_heights(amp, width*1e-6) / backg
            #print(snr)
            #red_split = width / split
            #width_tot.append(width)
            #freqs.append(frequency / numax[i])
            if cut == "sims":
                if (width >= 0.00787 * 1.0) and (snr > 18.9): # and (errs[-1]/inc < 0.3):
                    try:
                         samples = np.loadtxt(dirs[j], usecols=(5,))
                    except:
                         samples = np.loadtxt(dirs[j])
                    # If we want informatino about modes
                    if output_type == "modes":
                        try:
                            tmp = compute_values(samples, 0.683)
                            # Not sure what I'm doing here!
                            if tmp[-1] < 150.0:
                                #samples_tot.append(samples)
                                samples_tot.append(np.random.choice(samples, 1000, replace=False))
                                sigma_mode = np.append(sigma_mode, tmp[-1])
                                inc_modes = np.append(inc_modes, tmp[0])
                            else:
                                pass
                        except:
                            pass
                    elif output_type == "stars":
                        # Make a note of median inclination angle and hpd
                        tmp = compute_values(samples, 0.683)

                        average_inc = np.append(average_inc, np.median(samples))
                        average_error_pos = np.append(average_error_pos, tmp[1])
                        average_error_neg = np.append(average_error_neg, tmp[2])
                        # Take 1000 samples from each mode to combine into inclination
                        # posterior of star
                        samples_tot.append(np.random.choice(samples, 1000, replace=False))
                    # Counting number of modes per star
                    mds += 1
            else:
                pass

            #all_samples.append(np.random.choice(samples, 1000, replace=False))
        if output_type == "stars":
            # Flatten nested list of samples
            samples_tot = flatten(samples_tot)

            # Randomly choose 1000 samples from concatenated posterior to use in
            # hierarchical inference
            try:
                all_samples.append(np.random.choice(samples_tot), 1000, replace=False)
                n_modes = np.append(n_modes, mds)
            except:
                pass
        elif output_type == "modes"
            tmp = compute_values(samples_tot, 0.683)
            sigma_posts = np.append(sigma_posts, tmp[-1])
            inc_tot = np.append(inc_tot, tmp[0])

    if output_type == "stars":
        return all_samples
    elif output_type == "modes"
        return sigma_mode, sigma_posts, inc_modes, inc_tot



if __name__ == "__main__":
    """
    TODO: Add kepmag and red split cuts
    """
    # Create folder to store data and plots in

    # Artificial data
    #fname = sys.argv[1]
    #list_dirs = np.loadtxt(fname, usecols=(0,), dtype=float)
    #list_dirs = list_dirs.astype(int)
    #list_dirs = np.linspace(0, 89, 90).astype(int).astype(str)
    os.getcwd()
    # Real data
    rootdir = '/home/jsk389/Dropbox/Python/Peak-Bagging/Red_Giants/'
    rootdir = '/home/jsk389/Dropbox/Python/Peak-Bagging/Red_Giants/Tom/knownseismiccases/'
    #rootdir = '/home/jsk389/Dropbox/Python/Artificial-red-giant-power-spectra/Bespoke_new_widths/Isotropic_prior/'
    #numax = np.loadtxt('../../Artificial-red-giant-power-spectra/artif_numax.txt')
    list_dirs = ['8564976']


    output_type = "stars"
    all_samples = extract_data = (root_dir, list_dirs,
                                  cut='sims', output_type='stars')
