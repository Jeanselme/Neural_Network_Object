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

	void compute(int tid) {
		next->addSum(weight * previous->getResult(tid), tid);
	};

	void back(double learning_rate, int tid) {
		// In order to avoid aoverfitting and too large weight, we add a regularization term
		// This one has to be positive
		newWeight[tid] += next->getDelta(tid) * previous->getResult(tid) * learning_rate;
		previous->addDelta(weight * next->getDelta(tid), tid);
	};

	void update(double learning_rate, double regularization) {
		for (int i = 0; i < OMP_NUM_THREADS; i++) {
			weight -= newWeight[i];
			newWeight[i] = 0;
		}
		weight -= learning_rate * regularization * weight;
	};

	double getWeight() {
		return weight;
	};

	void setWeight(double nw) {
		weight = nw;
	};
};

#endif
