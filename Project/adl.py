"""
AUTHOR	: TAN WEI JIE AMOS
EMAIL	: amos.tan.2014@sis.smu.edu.sg

Description: 
This script plots the distribution of non-activity for each elder for varying time frames.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#function to read csv
def read(filename='../Workspace/DATA/SensorReading_2015-10_S001.csv'):
	df = pd.read_csv(filename, delimiter=',', usecols=[x for x in range(0,8)],parse_dates=[1])
	return df

df = read()
column_names = list(df.columns.values)
print(column_names)
print(len(df.index))
another_df = df.ix[(df['door_contact_as'] == 'No') & (df['living_room_as'] == 'No') & (df['bedroom_as'] == 'No') & (df['bed_as'] == 'No') & (df['bathroom_as'] == 'No') & (df['kitchen_as'] == 'No')]

print(len(another_df.index))
another_df.to_csv('test.csv')



