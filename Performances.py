import os

NUM_THREADS = 8

for i in range(1,NUM_THREADS+1):
	print(" ====== Executing with : " + str(i) + " threads ====== ")
	os.system("make clean run OMP_NUM_THREADS=" + str(i))
