"""TEST"""

import numpy as np
import scipy as sp


csv = np.genfromtxt('DATA/SensorReading_2015-10_S001.csv', dtype=str, delimiter=",")
print(csv)
