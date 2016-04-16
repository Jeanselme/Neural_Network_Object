#ifndef __NODE_H
#define __NODE_H
#include "activation.h"

class Neuron {
	/**
	 * Class for representing a perceptron
	 **/
protected:
	//
	double sumPrevious;
	// 
	double delta;
public:
	Neuron() {
		sumPrevious = 0;
		delta = 0;
	};
	
	double getResult() {
		return functionSigmoid(sumPrevious);
	};

	double getDelta() {
		return derivativeSigmoid(sumPrevious)*delta;
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