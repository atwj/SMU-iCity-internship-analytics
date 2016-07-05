"""
AUTHOR  : TAN WEI JIE AMOS
EMAIL   : amos.tan.2014@sis.smu.edu.sg
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

    return X,input_rad