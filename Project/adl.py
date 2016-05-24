"""
AUTHOR	: TAN WEI JIE AMOS
EMAIL	: amos.tan.2014@sis.smu.edu.sg
DATE    : 

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

time_now = time.time()
#function to read csv
def read(filename='../Workspace/DATA/SensorReading_2015-10_S001.csv'):
    df = pd.read_csv(filename, delimiter=',', usecols=[x for x in range(0,8)],parse_dates=[1])
    return df

#get dataframe containing readings from sensor reading
df = read()
column_names = list(df.columns.values)

#extract 'no-activity' periods
df = df.ix[(df['door_contact_as'] == 'No') & (df['living_room_as'] == 'No') 
                   & (df['bedroom_as'] == 'No') & (df['bed_as'] == 'No') 
                   & (df['bathroom_as'] == 'No') & (df['kitchen_as'] == 'No')]
#reset index
df = df.reset_index(drop=True)

#get difference between rows
df['TimeDelta'] = pd.TimedeltaIndex(df['date'].diff().fillna(0))
time_delta = df['TimeDelta']
print(list(df))
time_stamp = df['date']


print(type(time_delta))
print(type(time_stamp))

time_dict = []
some_time = time.time()

startTime = None
prevTime = None
for timestamp, delta in zip(time_stamp, time_delta):
    startTime = timestamp if startTime == None else startTime
    if delta.total_seconds() > 10:
        time_dict.append((startTime, (prevTime - startTime).total_seconds()))
        startTime = timestamp
        prevTime = timestamp
    else:
        prevTime = timestamp
        
new_df = pd.DataFrame(time_dict, columns=['timestamp','duration'])

print(new_df.head())
print()
print(new_df.describe())
print("Elasped Time: ", round(time.time() - time_now, 3), "seconds")