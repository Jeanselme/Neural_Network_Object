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
		printf("%f - ", (neurons.at(i))->getResult());
		if ((neurons.at(i))->getResult() > (neurons.at(max_neuron))->getResult()) {
			max_neuron = i;
		}
		if (labels.at(i) > labels.at(max_label)){
			max_label = i;
		}
	}
	printf("> %d -- %d\n", max_neuron, max_label);
	return max_neuron == max_label;
}

int main() {
	vector< vector<double> > images;
	vector< vector<int> > labels;
	string database = "../Python/Initial/train-images.idx3-ubyte";
	string labelname = "../Python/Initial/train-labels.idx1-ubyte";
  int inputDimension = readMNIST(database.c_str(), labelname.c_str(), images, labels);

	Network* net = new Network(NUMBER_LAYER);

	net->addInputs(inputDimension);
	Bias* b0 = new Bias();
	net->addNode(b0,0);

	net->addNodes(200,1);
	net->fullLinkage(0,1);
	Bias* b1 = new Bias();
	net->addNode(b1,1);

	net->addNodes(1,2);
	net->fullLinkage(1,2);

	net->backpropagation(images, labels);

	database = "../Python/Initial/t10k-images.idx3-ubyte";
	labelname = "../Python/Initial/t10k-labels.idx1-ubyte";
  inputDimension = readMNIST(database.c_str(), labelname.c_str(), images, labels);

  int correct = 0;
	for (uint i = 0; i < labels.size(); ++i) {
		net->compute(images.at(i));
		if (indice(net->getResult(), labels.at(i))) {
			correct ++;
		}
	}
	printf("%d / %d -> Classified\n", correct, (int) labels.size());
}
