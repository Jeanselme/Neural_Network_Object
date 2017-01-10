#include "Extraction/extraction.h"
#include "Network/network.h"
#include "Network/node.h"
#include <stdio.h>
#include <string>

#define NUMBER_LAYER 3

bool indice(vector<Neuron*> neurons, vector<int> labels) {
	int max_neuron = 0;
	int max_label  = 0;
	for (uint i = 0; i < neurons.size(); ++i) {
		if ((neurons.at(i))->getResult() > (neurons.at(max_neuron))->getResult()) {
			max_neuron = i;
		}
		if (labels.at(i) > labels.at(max_label)){
			max_label = i;
		}
	}
	return max_neuron == max_label;
}

int main() {
	vector< vector<double> > images;
	vector< vector<int> > labels;
	string database = "Data/train-images-idx3-ubyte";
	string labelname = "Data/train-labels-idx1-ubyte";
	int inputDimension = readMNIST(database.c_str(), labelname.c_str(), images, labels);

	Network net = Network(NUMBER_LAYER);

	net.addInputs(inputDimension);
	Bias* b0 = new Bias();
	net.addNode(b0,0);

	net.addNodes(NETWORK,1);
	net.fullLinkage(0,1);
	Bias* b1 = new Bias();
	net.addNode(b1,1);

	net.addNodes(10,2);
	net.fullLinkage(1,2);

	net.backpropagation(images, labels);

	database = "Data/t10k-images-idx3-ubyte";
	labelname = "Data/t10k-labels-idx1-ubyte";
	inputDimension = readMNIST(database.c_str(), labelname.c_str(), images, labels);

	#if TIME == 1
	double start_time, run_time;
	start_time = omp_get_wtime();
	#endif

	int correct = 0;
	for (uint i = 0; i < labels.size(); ++i) {
		net.computeParallel(images.at(i));
		if (indice(net.getResult(), labels.at(i))) {
			correct ++;
		}
	}
	#if TIME == 1
	run_time = omp_get_wtime() - start_time;
	printf("\n Testing in %lf seconds by image\n",run_time);
	#endif
	printf("%d / %d -> Classified\n", correct, (int) labels.size());
}
