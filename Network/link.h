#ifndef __LINK_H
#define __LINK_H

#include <stdlib.h>
#include <stdio.h>
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
	double newWeight;

public:
	Link(Neuron* n1, Neuron* n2) {
		Random *random = Random::get();
		previous = n1;
		next = n2;
		weight = random->getRandom() * 2 - 1;
		newWeight = weight;
	};

	void compute() {
		next->addSum(weight * previous->getResult());
	};

	void back(double learning_rate, double regularization) {
		// In order to avoid aoverfitting and too large weight, we add a regularization term
		// This one has to be positive
		newWeight -= next->getDelta() * previous->getResult() * learning_rate;
		previous->addDelta(weight * (next->getDelta() + regularization));
	};

	void update() {
		weight = newWeight;
	};

	double getWeight() {
		return weight;
	};
};

#endif
