from numba import jit
import numpy as np
from multiprocessing import Process as P
from itertools import combinations

objs = np.array([])

#c'vo
#public static dict MakeTree(lst):
	#gebruiken we denk ik niet bij ellipsbanen?
#	return 1

for tijdstap:
	pairs = np.array([i for i in combinations(objs,2)])