"""
AUTHOR	: TAN WEI JIE AMOS
EMAIL	: amos.tan.2014@sis.smu.edu.sg
DATE    : 
"""

###########################################IMPRORTS#########################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from math import pi
from sklearn.cluster import DBSCAN
###########################################IMPRORTS#########################################

# To return value in mins / total mins in one day
def to_mins(x):
    x = pd.Timestamp(x)
    year = x.year
    month =  x.month
    day = x.day
    return (x.value - pd.Timestamp(str(year)+'-'+str(month)+'-'+str(day)).value) / (60 * (10**9))

# Helper method to convert values to radian
def convert_to_radian(x):

    return ((x / (24*60)) * 2 * pi)

# Get all the sleep intervals based on bedroom only data
def get_df_sleep_intervals(df):
    
    df_sleep = pd.DataFrame(columns=('sleep_start', 'sleep_end', 'sleep_duration'))
    
    sleep_start = None
    sleep_end = None
    
    for index, row in df.iterrows():
        if (row['bedroom_only'] and sleep_start is None):
            sleep_start = row['date']
        elif ((not row['bedroom_only']) and (not sleep_start is None)):
            sleep_end = row['date']
            sleep_duration = sleep_end - sleep_start
        
            payload = {}
            payload['sleep_start'] = sleep_start
            payload['sleep_end'] = sleep_end
            payload['sleep_duration'] = sleep_duration
            df_sleep = df_sleep.append(payload, ignore_index=True)
        
            sleep_start = None
            sleep_end = None
            
    # already reached end of file, sleep was spilled over from last day to next month
    if not sleep_start is None:        
        sleep_end = df['date'].iloc[-1]
        sleep_duration = sleep_end - sleep_start
        
        payload = {}
        payload['sleep_start'] = sleep_start
        payload['sleep_end'] = sleep_end
        payload['sleep_duration'] = sleep_duration
        df_sleep = df_sleep.append(payload, ignore_index=True)
        
        sleep_start = None
        sleep_end = None
          
    # TODO: handle first and last entry in a more appropriate manner 
    # df_sleep.drop(df_sleep.head(1).index, inplace=True)
        
    # print(df_sleep.head())  
    # print(df_sleep.tail())  
    return df_sleep

# Returns a distance matrix (a numpy array)
def get_x_from_df(series):
    print(series.head())
    
    # Vectorizing to_mins and to_radian functions
    tmin = np.vectorize(to_mins)
    trad = np.vectorize(convert_to_radian)

    # Converting series of timestamp -> minutes / total minuites in a day -> radian
    input_rad = trad(tmin(series))

    # Convert time to rad points   
    X = input_rad[None,:] - input_rad[:,None]

    # Assign 'shortest distance to each point
    X[((X > pi) & (X <= (2*pi)))] = X[((X > pi) & (X <= (2*pi)))] -(2*pi)
    X[((X > (-2*pi)) & (X <= (-1*pi)))] = X[((X > (-2*pi)) & (X <= (-1*pi)))] + (2*pi) 
    X = abs(X)

    return X

# Returns output from sklearn DBSCAN method
def dbscan(eps, min_pts, metric, X):
    db = DBSCAN(eps, min_pts, metric)
    db.fit(X)
    return db.labels_, db.components_, db.core_sample_indices_

# TODO:
# 1. Research on methods to identify EPS.
# 2. Implement that method
def calculate_eps():
    pass

############################################################################################

# Start timer:
time_now = time.time()

### Globals ###
file_dir = 'data'
file_name = ''.join([file_dir,'/','SensorReading_2015-10_S001.csv'])

# print file name
print('File name: ', file_name)

# Get dataframe containing readings from sensor reading, exclude 
df = pd.read_csv(file_name, delimiter=',', usecols=[x for x in range(0,8)],parse_dates=[1])
column_names = list(df.columns.values)
# print('Before preprocessing: ', df.head(), df.tail())

#Remove rows where all sensors report 'no'
df = df.ix[(df['door_contact_as'] == 'Yes') | (df['living_room_as'] == 'Yes') 
	| (df['bedroom_as'] == 'Yes') | (df['bed_as'] == 'Yes') 
	| (df['bathroom_as'] == 'Yes') | (df['kitchen_as'] == 'Yes')]

# Add new column with bedroom_only flag
# TODO: handle for other cases where more sensors are valid
bedroom_only = ((df['door_contact_as'] == 'No') &
                (df['living_room_as'] == 'No') & 
                (df['bedroom_as'] == 'Yes') & 
                (df['bathroom_as'] == 'No') & 
                (df['kitchen_as'] == 'No'))

# Assign Series to dataframe.
df['bedroom_only'] = bedroom_only
# df = df.reset_index(drop=True)
df_sleep = get_df_sleep_intervals(df)
# print(df_sleep.columns.values)
# print(df_sleep.head())

# Set 'X1' as sleep_start timings
X1 = get_x_from_df(df_sleep['sleep_start'])

# Set 'X2' as sleep_end timings
X2 = get_x_from_df(df_sleep['sleep_end'])

# TODO
# 1. After implementing method to calculate EPS
# 2. Get output for both X1 and X2
# 3. Construct cluster plot. Polar plot
# X1_label, X1_components, X1_csi = dbscan()


# trad = np.vectorize(convert_to_radian)

# print('After preprocessing: ', df.head(), df.tail())
# Reset index
# df = df.reset_index(drop=True)

#get difference between timestamps 
# df['TimeDelta'] = df['date'].diff().fillna(0)
# df['in_mins'] = np.array(df['TimeDelta'] / np.timedelta64(1, 'm'))

#'sleep' time > 1 min
# df = df.ix[(df['in_mins'] > 1)]
# df = df.reset_index(drop=True)

# print(' Statistics for df["in_mins"] \n', df['in_mins'].describe()) 

#plot
# # x = df['in_mins']
# binwidth = 30
# bins = range(int(min(x)), int(max(x) + binwidth), binwidth)
# plt.hist(x, bins)
# plt.xlabel('duration')
# plt.ylabel('freq')
# plt.show()

#preprocessing, change timestamp value in minutes
# tmin = np.vectorize(to_mins)
# trad = np.vectorize(convert_to_radian)

# #DBSCAN
# input_ = tmin(df['date'])
# input_rad = trad(input_)

# #convert time to rad points
# X = input_rad[None,:] - input_rad[:,None]
# #assign 'shortest distance to each point'
# X[((X > pi) & (X <= (2*pi)))] = X[((X > pi) & (X <= (2*pi)))] -(2*pi)
# X[((X > (-2*pi)) & (X <= (-1*pi)))] = X[((X > (-2*pi)) & (X <= (-1*pi)))] + (2*pi) 
# X = abs(X)

# eps = 72.750858
# eps_rad = (eps / (24*60)) * 2 * pi 
# min_samples = 1

# # Set DBSCAN parameters
# db = DBSCAN(eps_rad, min_samples, metric='precomputed')

# # fit estimator
# db.fit(X)

# #get labels
# labels = db.labels_
# print(labels)
# # total there are n clusters + noise in labels. 
# # - 1 if -1 exist in labels because -1 is used to denote noise
# no_clusters = len(set(labels)) - (1 if -1 in labels else 0)
# components = db.components_
# #csi = core sample indices
# csi = db.core_sample_indices_

# # Polar plot

# plt.polar(2*pi, 1)
# # plt.show()
# print('No. Of Clusters: ', no_clusters)

print("Elasped Time: ", round(time.time() - time_now, 3), "seconds")