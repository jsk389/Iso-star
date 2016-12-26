#!/usr/bin/python

from __future__ import division

import numpy as np

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
