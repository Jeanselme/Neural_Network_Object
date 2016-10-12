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

public:
	Link(Neuron* n1, Neuron* n2) {
		Random *random = Random::get();
		previous = n1;
		next = n2;
		weight = random->getRandom()*2 - 1;
	};

	void compute() {
		next->addSum(weight * previous->getResult());
	};

	void back(double learning_rate) {
		weight -= next->getDelta() * previous->getResult() * learning_rate;
		previous->addDelta(weight * next->getDelta());
	};

	double getWeight() {
		return weight;
	};
};

#endif
