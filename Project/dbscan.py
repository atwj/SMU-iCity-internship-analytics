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
from math import pi

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
	# return visited,noise

	# Initialize and convert 'convert_to_radian' into numpy function
	rad_function = np.vectorize(convert_to_radian)

	# Convert points to radian
	points_rad = rad_function(points)

	# Convert eps to radian
	eps_rad = rad_function(eps).max()

	# Pre-compute distance from every point to every other point
	# O(N^2)
	dist = points_rad[None,:] - points_rad[:,None]
	# print(dist)

	#Assign shortest distances
	dist[((dist > pi) & (dist <= (2*pi)))] = dist[((dist > pi) & (dist <= (2*pi)))] -(2*pi)
	dist[((dist > (-2*pi)) & (dist <= (-1*pi)))] = dist[((dist > (-2*pi)) & (dist <= (-1*pi)))] + (2*pi) 
	dist = abs(dist)
	# print(dist)

	# DBSCAN Algorithm
	for i in range(len(points_rad)):
		if visited[i] == 1:
			continue

		visited[i] = 1
		neighbors = get_neighbors(i, dist, eps_rad)

		if (sum(neighbors) < min_points):
			noise[i] = 1
		else:
			cluster_index = cluster_index + 1
			new_cluster, neighbors_visited = expand_cluster(i, neighbors, 
				visited, dist, eps_rad, min_points, cluster)
			cluster[cluster_index, 1] = new_cluster
			visited = neighbors_visited


def get_neighbors(index, dist, eps):
	neighbors = dist[index]
	neighbors[(neighbors > eps)] = -1
	neighbors[(neighbors != -1)] = 1
	neighbors[(neighbors == -1)] = 0
	return neighbors # EXPECTED: 1-D array

def exists_in_cluster(index, cluster):
	exists = false
	for i in len(cluster):
		j = cluster[i]
		if sum(j == index) > 0:
			exists = true
			return exists
	return exists # EXPECTED bool val

def expand_cluster(index, neighbors, visited, dist, eps, min_points, cluster):
	new_cluster = index
	k = np.nonzero(neighbors)[0]
	j = 0 # Python arrays start from 0, matlab starts from 1

	while j < len(k):
		neighbor_index = k[j]
		if visited[neighbor_index] != 1:
			visited[neighbor_index] = 1
			next_neighbors = get_neighbors(neighbor_index, dist, eps)
			if sum(next_neighbors) >= min_points:
				pass
	return

def convert_to_radian(x):
	return ((x / (24*60)) * 2 * pi)