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
import math
from sklearn.cluster import DBSCAN
###########################################IMPRORTS#########################################

#Start timer:
time_now = time.time()

### Globals ###
file_dir = 'data'
file_name = ''.join([file_dir,'/','SensorReading_2015-10_S001.csv'])

# print file name
print('File name: ', file_name)

#get dataframe containing readings from sensor reading
df = pd.read_csv(file_name, delimiter=',', usecols=[x for x in range(0,8)],parse_dates=[1])
column_names = list(df.columns.values)

# Bedroom sensor indicates movement and the rest of the house indicates no movement
df = df.ix[(df['door_contact_as'] == 'No') & (df['living_room_as'] == 'No') 
                   & (df['bedroom_as'] == 'Yes') & (df['bed_as'] == 'No') 
                   & (df['bathroom_as'] == 'No') & (df['kitchen_as'] == 'No')]

#reset index
df = df.reset_index(drop=True)

#get difference between timestamps 
df['TimeDelta'] = df['date'].diff().fillna(0)
df['in_mins'] = np.array(df['TimeDelta'] / np.timedelta64(1, 'm'))
# check data type; expected 'timedetla'
print('Check type : ', type(df['in_mins'][0]))
print(' Statistics for df["in_mins"] ', '/n' , df['in_mins'].describe()) 

#plot
df['in_mins'].hist()

# time_dict = []
# some_time = time.time()

# startTime = None
# prevTime = None
# for timestamp, delta in zip(time_stamp, time_delta):
#     startTime = timestamp if startTime == None else startTime
#     if delta.total_seconds() > 10:
#         time_dict.append((startTime, (prevTime - startTime).total_seconds()))
#         startTime = timestamp
#         prevTime = timestamp
#     else:
#         prevTime = timestamp
        
# new_df = pd.DataFrame(time_dict, columns=['timestamp','duration'])

# #extract day of week

# #extract hour of day

# #extract minutes 

# #extracts seconds 


# # print(type(new_df['timestamp']))
# print(new_df, '\n')

# # new_df['timestamp'] = new_df['timestamp'].map(pd.Timestamp.time)
# # new_df['timestamp'] = new_df['timestamp'].apply(lambda x: x.strftime('%H:%M:%S'))
# # new_df['timestamp'] = new_df['timestamp'].apply(lambda x: pd.to_datetime(x, ))

# #convert timestamp to unix time, in float
# print(new_df.describe())

# nd_array = new_df.as_matrix()
# print(len(nd_array), nd_array[0:5])
# db = DBSCAN().fit(nd_array)
# print(db, '\n')
# print(len(set(db.labels_)) - (1 if -1 in db.labels_ else 0))

print("Elasped Time: ", round(time.time() - time_now, 3), "seconds")