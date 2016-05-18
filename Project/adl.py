"""
AUTHOR	: TAN WEI JIE AMOS
EMAIL	: amos.tan.2014@sis.smu.edu.sg
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
print(column_names)
print(len(df.index))

#extract 'no-activity' periods
df = df.ix[(df['door_contact_as'] == 'No') & (df['living_room_as'] == 'No') 
                   & (df['bedroom_as'] == 'No') & (df['bed_as'] == 'No') 
                   & (df['bathroom_as'] == 'No') & (df['kitchen_as'] == 'No')]


#testing code; remove later: print size of 
print(len(df.index))
df = df.reset_index(drop=True)
time_dict = {} 

for index, row in enumerate(df.itertuples()):
    startTimeIndex = index if index == 0 else startTimeIndex
    timestamp = row[2]
    if index + 1 < len(df.index):
        #check if current timestamp and next timestamp are <= 10 second time difference
        if (df.at[index + 1, 'date'] - timestamp).total_seconds() > 10:
            startTime = df.at[startTimeIndex,'date']
            time_dict[str(startTime)] = (timestamp - startTime).total_seconds()
            startTimeIndex = index + 1
    
new_df = pd.DataFrame.from_dict(time_dict, orient='index')
    
print(new_df.head())
print("Elasped Time: ", round(time.time() - time_now, 3), "seconds")