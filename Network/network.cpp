#include "network.h"

vector<Neuron*> Network::getResult() {
	return neurons.back();
}

void Network::addNode(Neuron* newNode, int layer) {
	neurons.at(layer).push_back(newNode);
}

void Network::addLink(Neuron* n1, Neuron* n2, int layer_inf) {
	Link* l = new Link(n1,n2);
	links.at(layer_inf).push_back(l);
	n2->addPrevious(l);
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
	// Inverse the different loops in order to allow an easier compilation afterwards
	for (vector<Neuron*>::iterator node2 = neurons.at(layer2).begin(); node2 != neurons.at(layer2).end(); ++node2) {
		for (vector<Neuron*>::iterator node1 = neurons.at(layer1).begin(); node1 != neurons.at(layer1).end(); ++node1) {
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

void Network::computeParallel(vector<double> &inputs) {
	resetSum();
	int i = 0;
	for (vector<double>::iterator input = inputs.begin(); input != inputs.end(); ++input) {
		(neurons.at(0).at(i))->addSum(*input);
		i ++;
	}
	#pragma omp parallel
	{
		for (vector< vector<Neuron*> >::iterator it = neurons.begin(); it < neurons.end(); ++it) {
			#pragma omp for
			for (vector<Neuron*>::iterator neuron = it->begin(); neuron < it->end(); ++neuron) {
				vector<Link*> links = (*neuron)->getPrevious();
				for (vector<Link*>::iterator link = links.begin(); link < links.end(); ++link) {
					(*link)->compute();
				}
			}
		}
	}
}

void Network::backLayer(int tid) {
	for (vector< vector<Link*> >::reverse_iterator it = links.rbegin(); it != links.rend(); ++it) {
		for (vector<Link*>::iterator link = it->begin(); link != it->end(); ++link) {
			(*link)->back(tid);
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

void shuffleInputs(vector< struct train_data > &inputs_targets, vector< vector<double> > &inputs, vector< vector<int> > &targets) {
	// This function shuffles the dataset
	inputs_targets.clear();
	vector <int> shuffle;
	for (int i=0; i<int(inputs.size()); i++) shuffle.push_back(i);
	random_shuffle(shuffle.begin(),shuffle.end());
	for (vector< int >::iterator order = shuffle.begin(); order != shuffle.end(); ++order) {
		struct train_data train = {inputs.at(*order), targets.at(*order)};
		inputs_targets.push_back(train);
	}
}

void Network::backpropagation(vector< vector<double> > &inputs, vector< vector<int> > &targets) {
	// Init for paralleliezation and displaying
	omp_set_num_threads(OMP_NUM_THREADS);
	cout << fixed << setprecision (2);

	// Different varaibles necessary for backpropagation
	double error = TOLERATE_ERROR; // Error of the iteration
	double pasterror = 10; // Error of the last iteration
	double learning_rate = 0.001;
	double regularization = 1/SIZE_BATCH;

	int tour = 1;
	int batch = inputs.size()/SIZE_BATCH; // Number of batch

	// Association of input and target for supervised learning
	vector< struct train_data > inputs_targets;

	#if TIME == 1
	double start_time, run_time;
	start_time = omp_get_wtime();
	#endif

	while (fabs(error - pasterror) >= TOLERATE_ERROR && tour <= MAX_ITERATION) {
		// Shuffles the dataset
		shuffleInputs(inputs_targets, inputs, targets);

		// Updates the learning rate
		if (pasterror < error) {
			learning_rate /= 2;
		}

		// Updates Error
		pasterror = error;
		error = 0;

		#if VERBOSE ==1
		int image = 0;

		printf("\nLearning -- %d\n", tour);
		#endif

		// Pragma outside in order to avoid to reinit the different threads
		#pragma omp parallel
		{
			int tid = omp_get_thread_num(); // Define the part in which the thread writes

			// Computes for each image the backpropagation
			for (int number_batch = 0; number_batch < batch; ++number_batch) {
				// Parallelize the image for a batch of images
				// Each thread will compute a part of the images batch in a "private" part
				// of each noode, which will be merged at the end of the batch
				#pragma omp for reduction(+:error)
				for (vector< struct train_data >::iterator data = inputs_targets.begin() + number_batch*SIZE_BATCH;
						data < min(inputs_targets.begin() + (number_batch + 1)*SIZE_BATCH, inputs_targets.end()); data++) {
					compute(data->input, tid);
					vector<int>::iterator targetOut = data->target.begin();
					for (vector< Neuron* >::iterator output = neurons.back().begin(); output != neurons.back().end(); ++output) {
						// Compute the error and its derivative -> Euclidean norm
						double delta = (*output)->getResult(tid) - *targetOut;
						error += 0.5*pow(delta, 2);

						// Add to the first (from end) hidden layer the computed derivative of error
						(*output)->addDelta(delta, tid);

						// Next target for next image
						targetOut ++;
					}

					// BackPropagate the error
					backLayer(tid);
					resetDelta(tid);

					#if VERBOSE == 1
					image ++;
					if (tid == 0) {
						float p = (float)image*100/inputs_targets.size();
						cout << "\r> " << p << "%" << flush;
					}
					#endif
				}

				// Merge the different thread subgradient
				#pragma omp single
				updateLayer(learning_rate, regularization);
			}
		}
		#if VERBOSE == 1
		printf("\r--> %f\n", error/inputs.size());
		#endif
		tour++;
	}
	#if TIME == 1
	run_time = omp_get_wtime() - start_time;
  printf("\n Training in %lf seconds\n",run_time);
	#endif
}


void  Network::save(const char* saveFile) {
	cout << fixed << setprecision(8);
	ofstream saver(saveFile);
	if (saver.is_open()) {
		// Writes number of layer
		saver << links.size();
		for (vector< vector<Link*> >::iterator it = links.begin(); it != links.end(); ++it) {
			// Writes number of link by layer
			saver << endl << it->size() << endl;
			for (vector<Link*>::iterator link = it->begin(); link != it->end(); ++link) {
				// Writes each weight
				saver << (*link)->getWeight() << " ";
			}
		}
		saver.close();
	} else {
		printf("[Error] Unable to save file\n");
	}
}

void  Network::load(const char* saveFile) {
	cout << fixed << setprecision(8);
	ifstream loader(saveFile);
	if (loader.is_open()) {
		// Reads number of layer
		double number_layer;
		loader >> number_layer;
		assert(number_layer == links.size());
		for (vector< vector<Link*> >::iterator it = links.begin(); it != links.end(); ++it) {
			// Reads number of link by layer
			double number_neuron;
			loader >> number_neuron;
			assert(number_neuron == it->size());
			for (vector<Link*>::iterator link = it->begin(); link != it->end(); ++link) {
				// Reads each weight
				double weight;
				loader >> weight;
				(*link)->setWeight(weight);
			}
		}
		loader.close();
	} else {
		printf("[Error] Unable to load file\n");
	}
}
