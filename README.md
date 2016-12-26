# Iso-star
Are stellar inclination angles isotropically distributed? Let's use asteroseismology of red giants to find out!

A common assumption is that the inclination angle of stars (the angle of the star's rotation axis to our line-of-sight) are isotropically distributed. In other words that there is no preferential direction or that we are not in a special position with respect to the stars in the sky.

Techniques to determine a star's inclination angle either rely on spectroscopic observations and make use of the rotation period, radius and vsini; or use photometry by fitting spot models to the data - enter asteroseismology.

The _Kepler_ mission has observed many thousands (over 13,000!) of red giants over its 4 year nominal mission. Using asteroseismology we can hope to derive the inclination angle of some of these stars (there are added complications that restrict the parameter space we can explore) and use these to derive the overall distribution of inclination angles.

To make inferences about the population inclination angle distribution we use the method described in [Hogg et al. (2010)](https://arxiv.org/abs/1008.4146) and use Hierarchical Bayesian modelling to do so. This repository contains the code used in this project.
