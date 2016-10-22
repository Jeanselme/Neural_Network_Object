#ifndef __NETWORK_H
#define __NETWORK_H

#include "link.h"
#include "node.h"
#include "random.h"
#include <vector>
#include <math.h>
#include <stdio.h>
#include <iomanip>
#include <iostream>
#include <assert.h>
#include <algorithm>

using namespace std;

#define TOLERATE_ERROR 0.00001
#define SIZE_BATCH 100
#define NUMBER_STO 10

class Network {
	/**
	 * Class for representing a perceptron
	 **/
protected:
	// Vector of neurons organised by layer
	vector< vector<Neuron*> > neurons;
	// Vector of links
	vector< vector<Link*> > links;

public:
	Network(int number_of_layer) {
		assert(number_of_layer > 1);
		neurons.resize(number_of_layer);
		links.resize(number_of_layer - 1);
	};

	vector<Neuron*> getResult();

	void addNode(Neuron* newNode, int layer);

	void addLink(Neuron* n1, Neuron* n2, int layer_inf);

	void addNodes(int number_of_neuron, int layer);

	void addInputs(int number_of_input);

	void fullLinkage(int layer1, int layer2);

	void resetSum();

	void resetDelta();

	void compute(vector<double> &inputs);

	void backLayer(double learning_rate);

	void updateLayer();

	void backpropagation(vector< vector<double> > &inputs, vector< vector<int> > &targets);

	void printRes();

	void printDelta();

	void printWeight();
};

#endif
