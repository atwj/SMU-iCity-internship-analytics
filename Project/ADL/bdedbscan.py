'''
This is the implementation of the BDE-DBSCAN algorithm from the following journal,

Karami, A., & Johansson, R. (2014). 
Choosing DBSCAN Parameters Automatically using Differential Evolution. 
International Journal of Computer Applications IJCA, 91(7), 
1-11. doi:10.5120/15890-5059

'''

def bde_dbscan(dataset, i_matrix):
	maxIter = 500
	nPop = 500
	# nVar = 

	bestSol = []

	