"""
file writer
"""

from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
import cluster_sleep_periods as csp

def sleep_to_cluster_aggregate(path,output_path):

	files = [f for f in listdir(path) if isfile(join(path, f))]
	file_count = len(files)
	files_with_error = []
	for f in files:
		month = f[16:23]
		id = f[24:28]
		try:
			output_dict = csp.return_clusters(path + f)
			output_dict['id'] = id
			output_dict['month'] = month
			df = pd.DataFrame(output_dict)
			print(df.columns)
			df = df[['id','month','cluster','centroid','std','variance','start_end']]
			df.to_csv(output_path+'cluster_'+f)
		except ValueError:
			file_count -= 1
			files_with_error.append(f)
	print('Files parsed successfully ', file_count)
	print('Files with errors: ', files_with_error)


