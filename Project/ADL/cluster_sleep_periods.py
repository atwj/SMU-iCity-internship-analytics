###########################################IMPRORTS#########################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import helper_adl as h
import eps_min_pts as eps_And_MinPts
from math import pi
from sklearn.cluster import DBSCAN
from matplotlib.backends.backend_pdf import PdfPages
###########################################IMPRORTS#########################################

# Returns output from sklearn DBSCAN method 
def dbscan(eps, min_pts, X, metric='precomputed'):
    db = DBSCAN(eps, min_pts, metric)
    db.fit(X)
    return db.labels_, db.components_, db.core_sample_indices_

# Code
# Start timer:
# time_now = time.time()

### Globals ###
file_dir = 'sleep'
# Perform clustering on files 1 to 10
# indexes = [str(x) for x in range(1,10)]
index = '9'

# file_name = ''.join([file_dir,'/','sleep-aggregate_2016-01_S001.csv'])
# print file name
# print('File name: ', file_name)

def return_clusters(file_name):
    # Get dataframe containing readings from sensor reading, exclude 
    df = pd.read_csv(file_name, delimiter=',', usecols=[x for x in range(1,7)],parse_dates=[1])
    # column_names = list(df.columns.values)
    # print(column_names)
    # X is a distance matrix.
    # Set 'X1' as sleep_start timings
    X1,X1_rad_series = h.get_x_from_df(df['sleep_start'])

    # Set 'X2' as sleep_end timings
    X2,X2_rad_series = h.get_x_from_df(df['sleep_end'])

    # TODO
    # 1. Implement method to calculate EPS
    # 2. Get output for both X1 and X2
    # 3. Construct cluster plot. Polar plot

    # Arbitrary eps and min_pts value:
    eps_X1, min_pts_X1 = eps_And_MinPts.knee_calculate_eps_minPts(X1)
    # print('eps: ', eps_X1 , ' minPts: ' , min_pts_X1)
    eps_X2, min_pts_X2 = eps_And_MinPts.knee_calculate_eps_minPts(X2)
    # print('eps: ', eps_X2 , ' minPts: ' , min_pts_X2)
    # eps = 0.31
    # min_pts = 29

    X1_label, X1_components, X1_csi = dbscan(eps_X1, min_pts_X1, X1)
    X2_label, X2_components, X2_csi = dbscan(eps_X2, min_pts_X2, X2)

    # - 1 if -1 exist in labels because -1 is used to denote noise
    X1_no_clusters = len(set(X1_label)) - (1 if -1 in X1_label else 0)
    print('Number of clusters for start sleep time: ', X1_no_clusters )
    # print('end sleep time cluster: ')
    X2_no_clusters = len(set(X2_label)) - (1 if -1 in X2_label else 0)
    print('Number of clusters for end sleep time: ', X2_no_clusters )

    # EXTRACTION OF CLUSTERS
    ###########################################
    output_dict = {}
    cl = []
    print(type(cl))
    sd = []
    var = []
    centroid = []
    start_end = []
    cluster_dict_ = h.extract_clusters(X1_label, X1_rad_series)
    cluster_dict_keys_ = cluster_dict_.keys()
    for i in cluster_dict_keys_:
        cluster = cluster_dict_[i]
        cl.append(np.asscalar(i))
        sd.append(h.format_mins(h.radian_to_mins(np.std(cluster))))
        var.append(h.format_mins(h.radian_to_mins(np.var(cluster))))
        centroid.append(h.format_mins(h.radian_to_mins(np.median(cluster))))
        start_end.append('start')

    cluster_dict_ = h.extract_clusters(X2_label, X2_rad_series)
    cluster_dict_keys_ = cluster_dict_.keys()
    for i in cluster_dict_keys_:
        cluster = cluster_dict_[i]
        cl.append(i)
        sd.append(h.format_mins(h.radian_to_mins(np.std(cluster))))
        var.append(h.format_mins(h.radian_to_mins(np.var(cluster))))
        centroid.append(h.format_mins(h.radian_to_mins(np.median(cluster))))
        start_end.append('end')

    output_dict['cluster'] = np.array(cl)
    output_dict['sd'] = np.array(sd)
    output_dict['var'] = np.array(var)
    output_dict['centroid'] = np.array(centroid)
    output_dict['start_end'] = start_end

    return output_dict

# print(return_clusters(file_name))
##################PLOTTING####################
# TODO: CLUSTER
# fig = plt.figure(figsize=(6,6))
# ax1 = configure_polar_plot(fig.add_subplot(111, projection='polar'))
# ax1.set_title(file_name)
# ax1.plot(X1_rad_series, [1 for x in X1_rad_series], 'mo')
# ax1.plot(X2_rad_series, [0.8 for x in X2_rad_series], 'y*')
# plot_clusters(X1_label, X1_rad_series, ax1, 0.8)
# plot_clusters(X2_label, X2_rad_series, ax1, 0.6)
# plt.show()



# print("Elasped Time: ", round(time.time() - time_now, 3), "seconds")
