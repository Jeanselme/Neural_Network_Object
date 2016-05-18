#ifndef __LINK_H
#define __LINK_H

#include <stdlib.h>
#include <stdio.h>
#include "node.h"

#define LEARNING_RATE 0.5

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
		previous = n1;
		next = n2;
		weight = (static_cast <float> (rand()) 
			/ static_cast <float> (RAND_MAX)) * 2 - 1;
	};

	void compute() {
		next->addSum(weight * previous->getResult());
	};

	void back() {
		weight -= next->getDelta() * previous->getResult() * LEARNING_RATE;
		previous->addDelta(weight * next->getDelta());
	};
};

#endif