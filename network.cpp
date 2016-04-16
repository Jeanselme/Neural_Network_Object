#include "network.h"

vector<Neuron*> Network::getResult() {
	return neurons.back();
}

void Network::addNode(Neuron* newNode, int layer) {
	neurons.at(layer).push_back(newNode);
}

void Network::addLink(Neuron* n1, Neuron* n2) {
	links.push_back(new Link(n1,n2));
}

void Network::addNodes(int number_of_neuron, int layer) {
	for (int i = 0; i < number_of_neuron; ++i) {
		addNode(new Neuron(), layer);
	}
}

void Network::addInputs(int number_of_input) {
	for (int i = 0; i < number_of_input; ++i) {
		addNode(new Input(), 0);
	}
}

void Network::fullLinkage(int layer1, int layer2){
	for (vector<Neuron*>::iterator node1 = neurons.at(layer1).begin(); node1 != neurons.at(layer1).end(); ++node1) {
		for (vector<Neuron*>::iterator node2 = neurons.at(layer1).begin(); node2 != neurons.at(layer1).end(); ++node2) {
			addLink(*node1, *node2);
		}
	}
}


void Network::resetSum() {
	for (vector< vector<Neuron*> >::iterator it = neurons.begin(); it != neurons.end(); ++it) {
		for (vector<Neuron*>::iterator node = it->begin(); node != it->end(); ++node) {
			(*node)->reinitSum();
		}
	}
}

void Network::resetDelta() {
	for (vector< vector<Neuron*> >::iterator it = neurons.begin(); it != neurons.end(); ++it) {
		for (vector<Neuron*>::iterator node = it->begin(); node != it->end(); ++node) {
			(*node)->reinitDelta();
		}
	}
}

void Network::compute(vector<double> inputs) {
	resetSum();
	int i = 0;
	for (vector<double>::iterator input = inputs.begin(); input != inputs.end(); ++input) {
		(neurons.at(0).at(i))->addSum(*input);
		i ++;
	} 
	for (vector<Link*>::iterator link = links.begin(); link != links.end(); ++link) {
		(*link)->compute();
	}
}

void Network::backpropagation(vector< vector<double> > inputs, vector< vector<double> > targets) {
	double error = TOLERATE_ERROR;
	while (error >= TOLERATE_ERROR) {
		for (vector< vector<double> >::iterator input = inputs.begin(); input != inputs.end(); ++input) {
			resetDelta();
			compute(*input);
			int number_output = 0;
			for (vector< Neuron* >::iterator output = neurons.back().begin(); output != neurons.back().end(); ++output) {
				double delta = (*output)->getResult() - targets.at(number_output).at(number_output);
				(*output)->addDelta(delta);
				number_output ++;
				error += pow(delta, 2);
			}

			for (vector<Link*>::iterator link = links.begin(); link != links.end(); ++link) {
				(*link)->back();
			}
		}
		printf("%f\n", error);
	}
}