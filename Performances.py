import subprocess
import re
import matplotlib.pyplot as plt

NUM_THREADS = 8
MAXITER = 10

times = []

batchSize = [100,500,1000,5000]
style = ['ro', 'bo', 'go', 'yo']

for batch in batchSize:
	timeBatch = []
	for i in range(1,NUM_THREADS+1):
		print(" ====== Executing with : " + str(i) + " threads ====== ")
		result = subprocess.check_output("make clean run VERBOSE=0 OMP_NUM_THREADS=" + str(i) + " BATCH=" + str(batch) + " MAXITER=" + str(MAXITER), shell=True).decode('utf-8')
		time = float(re.search(r"Training in (?P<time>\d+\.\d+) s",result).group('time'))/MAXITER
		timeBatch.append(time)
		print(time)

	times.append(timeBatch)
plt.figure(1)

# Training error
for i in range(len(batchSize)):
	plt.plot(range(1,NUM_THREADS+1), times[i], style[i])
plt.xlim(0, NUM_THREADS + 1)
plt.xlabel('Number of threads')
plt.ylabel('Time per iteration')
plt.title('Average training time')

plt.show()
