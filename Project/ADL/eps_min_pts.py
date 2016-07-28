"""
Calculate eps & minPts

"""
################ IMPORTS ##################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.cluster import DBSCAN
################ IMPORTS ##################

# TODO:
# 1. Significance of similarity in the data????????????????

# Using the 'knee' method of choosing an eps and min pts
def knee_calculate_eps_minPts(distance_matrix):
	# Flatten matrix
	X = distance_matrix.flatten()
	
	# Get non-zeros
	X_non_zeros_ = X.nonzero()
	# Mask out non-zeroes
	X_array = X[X_non_zeros_]
	# Generate histogram
	hist, bin_edges = np.histogram(X_array, density=False)
	# Arbitrarily select the knee, occasionally it falls in the 2nd or 3rd bin.
	#TODO: Use vector projection to select knee point.
	eps = bin_edges[1]

	count_minPts = []
	for distance in distance_matrix:
		count_minPts.append(len(np.where(distance <= eps)[0]))

	# Generate histogram for minpts
	hist_mp, bind_edges_mp = np.histogram(count_minPts, density=False)

	# Arbitrarily select the max bin_edge value.
	# TODO: Find a better way to select the minpts value
	minPts = np.median(bind_edges_mp)

	# Return eps and minPts
	return eps, minPts




