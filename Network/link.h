#ifndef __LINK_H
#define __LINK_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>
#include "node.h"
#include "random.h"

class Link {
	/**
	 * Link between two neurons
	 **/
private:
	Neuron* previous;
	Neuron* next;
	// Weight of the link
	double weight;
	double newWeight[OMP_NUM_THREADS];

public:
	Link(Neuron* n1, Neuron* n2) {
		Random *random = Random::get();
		previous = n1;
		next = n2;
		weight = random->getRandom() * 2 - 1;
		memset(newWeight, 0, OMP_NUM_THREADS);
	};

	void compute(int tid = 0) {
		next->addSum(weight * previous->getResult(tid), tid);
	};

	void back(int tid = 0) {
		// In order to avoid aoverfitting and too large weight, we add a regularization term
		// This one has to be positive
		newWeight[tid] += next->getDelta(tid) * previous->getResult(tid);
		previous->addDelta(weight * next->getDelta(tid), tid);
	};

	void update(double learning_rate, double regularization) {
		double update = 0;
		for (int i = 0; i < OMP_NUM_THREADS; i++) {
			update += newWeight[i];
			newWeight[i] = 0;
		}
		weight -= learning_rate * (regularization * weight + update);
	};

	double getWeight() {
		return weight;
	};

	void setWeight(double nw) {
		weight = nw;
	};
};

#endif
