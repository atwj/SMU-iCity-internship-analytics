"""
Test script for dbscan.py
"""

import numpy as np
import dbscan as db
from sklearn.cluster import DBSCAN 
from sklearn.preprocessing import StandardScaler

data = np.genfromtxt('data.csv', delimiter=',')
print("Data input size", len(data))
cluster = db.dbscan(data, 60, 7)

# print to file(s):
for i in cluster.keys():
	file_name = "cluster_" + str(i)
	X = np.array(cluster[i])
	np.savetxt(file_name + '.csv', X, delimiter=',')

"""
Testing with scikit-learn
"""
