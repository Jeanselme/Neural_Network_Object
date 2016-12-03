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

void Network::resetSum(int tid) {
	for (vector< vector<Neuron*> >::iterator it = neurons.begin(); it != neurons.end(); ++it) {
		for (vector<Neuron*>::iterator node = it->begin(); node != it->end(); ++node) {
			(*node)->reinitSum(tid);
		}
	}
}

void Network::resetDelta(int tid) {
	for (vector< vector<Neuron*> >::iterator it = neurons.begin(); it != neurons.end(); ++it) {
		for (vector<Neuron*>::iterator node = it->begin(); node != it->end(); ++node) {
			(*node)->reinitDelta(tid);
		}
	}
}

void Network::compute(vector<double> &inputs, int tid) {
	resetSum(tid);
	int i = 0;
	for (vector<double>::iterator input = inputs.begin(); input != inputs.end(); ++input) {
		(neurons.at(0).at(i))->addSum(*input, tid);
		i ++;
	}
	for (vector< vector<Link*> >::iterator it = links.begin(); it != links.end(); ++it) {
		for (vector<Link*>::iterator link = it->begin(); link != it->end(); ++link) {
			(*link)->compute(tid);
		}
	}
}

void Network::backLayer(double learning_rate, int tid) {
	for (vector< vector<Link*> >::reverse_iterator it = links.rbegin(); it != links.rend(); ++it) {
		for (vector<Link*>::iterator link = it->begin(); link != it->end(); ++link) {
			(*link)->back(learning_rate, tid);
		}
	}
}

void Network::updateLayer(double learning_rate, double regularization) {
	for (vector< vector<Link*> >::reverse_iterator it = links.rbegin(); it != links.rend(); ++it) {
		for (vector<Link*>::iterator link = it->begin(); link != it->end(); ++link) {
			(*link)->update(learning_rate, regularization);
		}
	}
}

void Network::backpropagation(vector< vector<double> > &inputs, vector< vector<int> > &targets) {
	omp_set_num_threads(OMP_NUM_THREADS);
	cout << fixed << setprecision (2);
	double error = TOLERATE_ERROR;
	double pasterror = 10;
	double learning_rate = 0.1;
	double regularization = 1/SIZE_BATCH;
	int tour = 1;
	int batch = inputs.size()/SIZE_BATCH;
	int batch_image = inputs.size()/batch;

	vector< struct train_data > inputs_targets;

	double start_time, run_time;
	start_time = omp_get_wtime();

	while (fabs(error - pasterror) >= TOLERATE_ERROR && tour <= MAX_ITERATION) {
		// Shuffles the dataset
		inputs_targets.clear();
		vector <int> shuffle;
		for (int i=0; i<int(inputs.size()); i++) shuffle.push_back(i);
		random_shuffle(shuffle.begin(),shuffle.end());
		for (vector< int >::iterator order = shuffle.begin(); order != shuffle.end(); ++order) {
			struct train_data train = {inputs.at(*order), targets.at(*order)};
			inputs_targets.push_back(train);
		}

		// Updates the learning rate
		// TODO : Wolfe conditions
		if (pasterror < error) {
			learning_rate /= 2;
		}

		pasterror = error;
		error = 0;
		int image = 0;

		printf("\nLearning -- %d\n", tour);
		// Computes for each image the backpropagation
		for (int number_batch = 0; number_batch < batch; ++number_batch) {
			#pragma omp parallel for reduction(+:error)
			for (vector< struct train_data >::iterator data = inputs_targets.begin() + number_batch*batch_image;
			 		data < inputs_targets.begin() + (number_batch + 1)*batch_image; data++) {
				int tid = omp_get_thread_num();
				compute(data->input, tid);
				vector<int>::iterator targetOut = data->target.begin();
				for (vector< Neuron* >::iterator output = neurons.back().begin(); output != neurons.back().end(); ++output) {
					double delta = (*output)->getResult(tid) - *targetOut;
					error += 0.5*pow(delta, 2);
					(*output)->addDelta(delta, tid);
					targetOut ++;
				}

				backLayer(learning_rate, tid);
				resetDelta(tid);

				image ++;
				if (tid == 0) {
					float p = (float)image*100/inputs_targets.size();
					cout << "\r> " << p << "%" << flush;
				}
			}
			updateLayer(learning_rate, regularization);
		}
		printf("\r--> %f\n", error);
		tour++;
	}

	run_time = omp_get_wtime() - start_time;
  printf("\n Training in %lf seconds\n",run_time);
}
