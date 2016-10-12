#ifndef __NODE_H
#define __NODE_H

#include "activation.h"

class Neuron {
	/**
	 * Class for representing a perceptron
	 **/
protected:
	// Sum which will be computed in order
	// to obtain the output of the neuron
	double sumPrevious;
	// Delta of the neuron
	double delta;
public:
	Neuron() {
		sumPrevious = 0;
		delta = 0;
	};

	virtual double getResult() {
		return functionSigmoidUpdate(sumPrevious);
	};

	double getDelta() {
		return derivativeSigmoidUpdate(sumPrevious)*delta;
	};

	void addDelta(double delta_to_add) {
		delta += delta_to_add;
	};

	void addSum(double val_to_add) {
		sumPrevious += val_to_add;
	};

	void reinitDelta() {
		delta = 0;
	};

	void reinitSum() {
		sumPrevious = 0;
	};

};

class Bias : public Neuron {
	/**
	 * Particular perceptron which does not compute
	 **/
public:
	double getResult() {
		return 1;
	};
};

class Input : public Neuron {
	/**
	 * Particular perceptron which does not change
	 **/
public:
	double getResult() {
		return sumPrevious;
	};
};


#endif
