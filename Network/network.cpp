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

void Network::dropHalfNode(int layer){
	Random *random = Random::get();
	int nodesNumber = neurons.at(layer).size();
	bool toDrop[nodesNumber];
	random->randomBoolean(nodesNumber, toDrop, DROP_FRACTION);
	// Update Sum
	int i = 0;
	for (vector<Neuron*>::iterator node = neurons.at(layer).begin(); node != neurons.at(layer).end(); ++node) {
		if (toDrop[i]) {
			(*node)->reinitSum();
		} else {
			(*node)->multiplySum(1./DROP_FRACTION);
		}
		i++;
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

void Network::compute(vector<double> &inputs, bool dropout) {
	resetSum();
	int i = 0;
	for (vector<double>::iterator input = inputs.begin(); input != inputs.end(); ++input) {
		(neurons.at(0).at(i))->addSum(*input);
		i ++;
	}
	i = 1;
	for (vector< vector<Link*> >::iterator it = links.begin(); it != links.end(); ++it) {
		for (vector<Link*>::iterator link = it->begin(); link != it->end(); ++link) {
			(*link)->compute();
		}
		if (dropout) {
			dropHalfNode(i);
		}
		i++;
	}
}

void Network::backLayer(double learning_rate, double regularization) {
	for (vector< vector<Link*> >::reverse_iterator it = links.rbegin(); it != links.rend(); ++it) {
		for (vector<Link*>::iterator link = it->begin(); link != it->end(); ++link) {
			(*link)->back(learning_rate, regularization);
		}
	}
}

void Network::updateLayer() {
	for (vector< vector<Link*> >::reverse_iterator it = links.rbegin(); it != links.rend(); ++it) {
		for (vector<Link*>::iterator link = it->begin(); link != it->end(); ++link) {
			(*link)->update();
		}
	}
}

void Network::backpropagation(vector< vector<double> > &inputs, vector< vector<int> > &targets, bool dropout) {
	cout << fixed << setprecision (2);
	double error = TOLERATE_ERROR;
	double pasterror = 10;
	double learning_rate = 0.1;
	double regularization = 1/SIZE_BATCH;
	int tour = 1;
	int batch = inputs.size()/SIZE_BATCH;

	vector< vector<double> > inputs_studied;
	vector< vector<int> > targets_studied;

	while (fabs(error - pasterror) >= TOLERATE_ERROR && tour <= MAX_ITERATION) {
		// Shuffles the dataset
		inputs_studied.clear();
		targets_studied.clear();
		vector <int> shuffle;
		for (int i=0; i<int(inputs.size()); i++) shuffle.push_back(i);
		random_shuffle(shuffle.begin(),shuffle.end());
		for (vector< int >::iterator order = shuffle.begin(); order != shuffle.end(); ++order) {
			inputs_studied.push_back(inputs.at(*order));
			targets_studied.push_back(targets.at(*order));
		}

		// Updates the learning rate
		// TODO : Wolfe conditions

		pasterror = error;
		error = 0;
		int image = 0;
		vector< vector<int> >::iterator target = targets_studied.begin();

		printf("\nLearning -- %d\n", tour);
		// Computes for each image the backpropagation
		for (vector< vector<double> >::iterator input = inputs_studied.begin(); input != inputs_studied.end(); ++input) {
			compute(*input, dropout);
			vector<int>::iterator targetOut = target->begin();
			for (vector< Neuron* >::iterator output = neurons.back().begin(); output != neurons.back().end(); ++output) {
				double delta = (*output)->getResult() - *targetOut;
				error += 0.5*pow(delta, 2);
				(*output)->addDelta(delta);
				targetOut ++;
			}

			backLayer(learning_rate, regularization);
			resetDelta();

			image ++;
			target ++;
			float p = (float)image*100/inputs_studied.size();
			cout << "\r> " << p << "%" << flush;

			// Updates if it is a batch learning
			if (image%batch == 0) {
				updateLayer();
			}
		}
		printf("\r--> %f\n", error);
		tour++;
	}
}

void Network::printRes() {
	for (vector< vector<Neuron*> >::iterator it = neurons.begin(); it != neurons.end(); ++it) {
		for (vector<Neuron*>::iterator node = it->begin(); node != it->end(); ++node) {
			printf("%f ", (*node)->getResult());
		}
		printf("\n - \n");
	}
}

void Network::printDelta() {
	for (vector< vector<Neuron*> >::iterator it = neurons.begin(); it != neurons.end(); ++it) {
		for (vector<Neuron*>::iterator node = it->begin(); node != it->end(); ++node) {
			printf("%f ", (*node)->getDelta());
		}
		printf("\n - \n");
	}
}

void Network::printWeight() {
	for (vector< vector<Link*> >::reverse_iterator it = links.rbegin(); it != links.rend(); ++it) {
		for (vector<Link*>::iterator link = it->begin(); link != it->end(); ++link) {
			printf("%f ", (*link)->getWeight());
		}
		printf("\n - \n");
	}
}
