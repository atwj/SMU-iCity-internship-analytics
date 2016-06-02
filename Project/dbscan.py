"""
AUTHOR	: 	TAN WEI JIE, AMOS
TITLE	: 	dbscan.py
EMAIL	:	amos.tan.2014@sis.smu.edu.sg
DESC	: 	Python implementation of DBSCAN clustering algorithm. For more information of DBSCAN
			please refer to https://en.wikipedia.org/wiki/DBSCAN

			The DBSCAN method implemented in this file has been modified to cluster across 
			periods; E.g 2359H and 0002H

"""

import numpy as np

def dbscan(points, eps, min_points):
	"""
	Input: 
	Points		:	A 1-D array of time in minutes
	Eps			:	Distiance in time to be considered 'close' (minutes)
	Min_points	:	Minimum number of points to be a dense region.
	"""
	# Initialize cluster
	cluster = {}
	cluster_index = 0

	# 'Visited' matrix
	visited = np.zeros((len(points),1))

	# 'Noise' matrix
	noise = np.zeros((len(points),1))
	return visited,noise


def get_neighbors(idnex, dist_mod, eps):
	return

def exists_in_cluster(index, cluster):
	return
