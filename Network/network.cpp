#include "network.h"

vector<Neuron*> Network::getResult() {
	return neurons.back();
}

void Network::addNode(Neuron* newNode, int layer) {
	neurons.at(layer).push_back(newNode);
}

void Network::addLink(Neuron* n1, Neuron* n2, int layer_inf) {
	links.at(layer_inf).push_back(new Link(n1,n2));
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
		for (vector<Neuron*>::iterator node2 = neurons.at(layer2).begin(); node2 != neurons.at(layer2).end(); ++node2) {
			addLink(*node1, *node2, layer1);
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

void Network::compute(vector<double> &inputs) {
	resetSum();
	int i = 0;
	for (vector<double>::iterator input = inputs.begin(); input != inputs.end(); ++input) {
		(neurons.at(0).at(i))->addSum(*input);
		i ++;
	} 
	for (vector< vector<Link*> >::iterator it = links.begin(); it != links.end(); ++it) {
		for (vector<Link*>::iterator link = it->begin(); link != it->end(); ++link) {
			(*link)->compute();
		}
	}
}

void Network::backpropagation(vector< vector<double> > &inputs, vector< vector<int> > &targets) {
	cout << fixed << setprecision (2);
	double error = 0;
	double pasterror = TOLERATE_ERROR;
	int tour = 0;
	while (fabs(pasterror - error) >= TOLERATE_ERROR && tour < 100) {
		pasterror = error;
		error = 0;
		tour++;
		int image = 0;

		printf("\nLearning -- %d\n", tour);
		vector< vector<double> > inputs_studied;
		vector< vector<int> > targets_studied;
		#ifdef STOCHASTIC
		for (int i = 0; i < NUMBER_STO; ++i) {
			int index = rand() % inputs.size();
			inputs_studied.push_back(inputs.at(index));
			targets_studied.push_back(targets.at(index));
		}
		#else
		inputs_studied = inputs;
		targets_studied = targets;
		#endif
		for (vector< vector<double> >::iterator input = inputs_studied.begin(); input != inputs_studied.end(); ++input) {
			resetDelta();
			compute(*input);
			int number_output = 0;
			for (vector< Neuron* >::iterator output = neurons.back().begin(); output != neurons.back().end(); ++output) {
				double delta = (*output)->getResult() - targets_studied.at(image).at(number_output);
				(*output)->addDelta(delta);
				number_output ++;
				error += pow(delta, 2);
			}

			for (vector< vector<Link*> >::reverse_iterator it = links.rbegin(); it != links.rend(); ++it) {
				for (vector<Link*>::iterator link = it->begin(); link != it->end(); ++link) {
					(*link)->back();
				}
			}

			image ++;
			float p = (float)image*100/inputs_studied.size();
			cout << "\r> " << p << "%" << flush;

		}
		printf("\r--> %f\n", error);
	}
}

void Network::print() {
	for (vector< vector<Neuron*> >::iterator it = neurons.begin(); it != neurons.end(); ++it) {
		for (vector<Neuron*>::iterator node = it->begin(); node != it->end(); ++node) {
			printf("%f ", (*node)->getResult());
		}
		printf("\n - \n");
	}
}