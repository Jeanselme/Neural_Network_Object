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
#include <fstream>

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

	// Adds a node to the network at the given layer
	void addNode(Neuron* newNode, int layer);

	// Adds a link between two nodes previously added in the network
	void addLink(Neuron* n1, Neuron* n2, int layer_inf);

	// Adds a node in the network
	void addNodes(int number_of_neuron, int layer);

	// Adds an input (first layer of the network)
	void addInputs(int number_of_input);

	// Fully links two layers
	void fullLinkage(int layer1, int layer2);

	// Resets all neurons
	void resetSum(int tid);

	// Resets delta (needed for backpropagation) of all neurons
	void resetDelta(int tid);

	// Computes the foreward phase
	void compute(vector<double> &inputs, int tid = 0);

	// Parallel computation
	void computeParallel(vector<double> &inputs, int tid = 0);

	// Computes the backpropagation of a layer by updating the weight for the thread tid
	void backLayer(double learning_rate, int tid);

	// Updates the weights by summing the different values computed by threads
	void updateLayer(double learning_rate, double regularization);

	// Backprop
	void backpropagation(vector< vector<double> > &inputs, vector< vector<int> > &targets);

	// Saves all the weight of a neural network in the given file
	void save(const char* saveFile);

	// Loads a given neural network into the current one, need same structure
	void load(const char* saveFile);
};

#endif
