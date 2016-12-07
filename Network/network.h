#ifndef __NETWORK_H
#define __NETWORK_H

#include "link.h"
#include "node.h"
#include "random.h"
#include <omp.h>
#include <vector>
#include <math.h>
#include <stdio.h>
#include <iomanip>
#include <iostream>
#include <assert.h>
#include <algorithm>

using namespace std;

#define TOLERATE_ERROR 0.00001
#define MAX_ITERATION 10
#define SIZE_BATCH 100

struct train_data {
	vector<double> input;
	vector<int> target;
};

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

	~Network() {
		for (vector< vector<Link*> >::iterator it = links.begin(); it != links.end(); ++it) {
			for (vector<Link*>::iterator link = it->begin(); link != it->end(); ++link) {
				delete(*link);
			}
		}
		for (vector< vector<Neuron*> >::iterator it = neurons.begin(); it != neurons.end(); ++it) {
			for (vector<Neuron*>::iterator node = it->begin(); node != it->end(); ++node) {
				delete(*node);
			}
		}
	};

	vector<Neuron*> getResult();

	void addNode(Neuron* newNode, int layer);

	void addLink(Neuron* n1, Neuron* n2, int layer_inf);

	void addNodes(int number_of_neuron, int layer);

	void addInputs(int number_of_input);

	void fullLinkage(int layer1, int layer2);

	void resetSum(int tid);

	void resetDelta(int tid);

	void compute(vector<double> &inputs, int tid = 0);

	void computeParallel(vector<double> &inputs, int tid = 0);
	
	void backLayer(double learning_rate, int tid);

	void updateLayer(double learning_rate, double regularization);

	void backpropagation(vector< vector<double> > &inputs, vector< vector<int> > &targets);

	void printRes();

	void printDelta();

	void printWeight();
};

#endif
