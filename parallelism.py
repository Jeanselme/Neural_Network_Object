import subprocess
import re
import matplotlib.pyplot as plt

NUM_THREADS = 4
MAXITER = 10

trainingTimes = []
testingTimes = []

batchSize = [100, 500, 1000, 5000, 10000]
networkSize = [50, 100, 200, 400]
style = ['ro-', 'bo-', 'go-', 'yo-', 'mo-']

assert(len(batchSize) <= len(style) and len(networkSize) <= len(style))

# Computes train performance
for batch in batchSize:
	timeTrainBatch = []
	for i in range(1,NUM_THREADS+1):
		print(" ====== Training with : " + str(i) + " threads with a batch : " +str(batch)+ " ====== ")
		result = subprocess.check_output("make clean run VERBOSE=0 OMP_NUM_THREADS=" + str(i) + " BATCH=" + str(batch) + " MAXITER=" + str(MAXITER), shell=True).decode('utf-8')
		train = float(re.search(r"Training in (?P<time>\d+\.\d+) s",result).group('time'))/(MAXITER)
		timeTrainBatch.append(train)
		print(train)

	trainingTimes.append(timeTrainBatch)

# Computes test performance
for network in networkSize:
	timeTestBatch = []
	for i in range(1,NUM_THREADS+1):
		print(" ====== Testing with : " + str(i) + " threads with a network : " +str(network)+ " ====== ")
		result = subprocess.check_output("make clean run VERBOSE=0 OMP_NUM_THREADS=" + str(i) + " MAXITER=0 NETWORK=" + str(network), shell=True).decode('utf-8')
		test = float(re.search(r"Testing in (?P<time>\d+\.\d+) s",result).group('time'))/10 # Division because 10 000 images and results in ms
		timeTestBatch.append(test)
		print(test)

	testingTimes.append(timeTestBatch)

plt.figure("Training")
for i in range(len(batchSize)):
	plt.plot(range(1,NUM_THREADS+1), trainingTimes[i], style[i], label=str(batchSize[i]))
plt.xlim(0, NUM_THREADS + 1)
plt.xlabel('Number of threads')
plt.ylabel('Time per iteration (s)')
plt.title('Average training time')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.figure("Testing")
for i in range(len(networkSize)):
	plt.plot(range(1,NUM_THREADS+1), testingTimes[i], style[i], label=str(networkSize[i]) + " nodes")
plt.xlim(0, NUM_THREADS + 1)
plt.xlabel('Number of threads')
plt.ylabel('Time per image (ms)')
plt.title('Average testing time')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.show()
