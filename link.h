#ifndef __LINK_H
#define __LINK_H
#include <stdlib.h>
#include "node.h"

#define LEARNING_RATE = 0.9
#define LEARNING_FORGET = 0.7

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
		weight =  static_cast <float> (rand()) 
				/ static_cast <float> (RAND_MAX); 
	};

	void compute() {
		next->addSum(weight * previous->getResult());
	};

	void back() {
		weight -= next->getDelta() * previous->getResult();
		previous->addDelta(weight * next->getDelta());
	};
};

#endif