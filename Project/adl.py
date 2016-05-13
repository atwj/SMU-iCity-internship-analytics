"""
AUTHOR	: TAN WEI JIE AMOS
EMAIL	: amos.tan.2014@sis.smu.edu.sg

Description: 
This script plots the distribution of non-activity for each elder for varying time frames.
"""

import pandas as pd
import matplotlib.pyplot as plt

#function to read csv
def read(filename='/Users/AMOS/Documents/PROJECTS/Internship/Workspace/DATA/SensorReading_2015-10_S001.csv'):
	# s_id,date,door_contact_as,living_room_as,bedroom_as,bed_as,bathroom_as,kitchen_as 
	#csv = np.genfromtxt(filename,dtype=str, delimiter=',',skip_header = 1, usecols=(0,1,2,3,4,5,6,7))
	#realized that using a dataframe would be better.
	df = pd.read_csv(filename, delimiter=',', usecols=[x for x in range(0,8)],parse_dates=[1])
	return df

df = read()
print(df.values)
