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

	// To avoid computation repetition
	double result;
	bool computed;

	double deltaResult;
	bool deltaComputed;

public:
	Neuron() {
		sumPrevious = 0;
		delta = 0;
		computed = false;
		deltaComputed = false;
	};

	virtual ~Neuron() {};

	virtual double getResult() {
		if (!computed) {
			result = functionSigmoid(sumPrevious);
			computed = true;
		}
		return result;
	};

	double getDelta() {
		if (!deltaComputed) {
			deltaResult = derivativeSigmoid(result)*delta;
			deltaComputed = true;
		}
		return deltaResult;
	};

	void addDelta(double delta_to_add) {
		delta += delta_to_add;
	};

	void addSum(double val_to_add) {
		sumPrevious += val_to_add;
	};

	void reinitDelta() {
		delta = 0;
		deltaComputed = false;
	};

	void reinitSum() {
		sumPrevious = 0;
		computed = false;
	};

};

class Bias : public Neuron {
	/**
	 * Particular perceptron which does not compute
	 **/
public:
	~Bias() {};
	double getResult() {
		return 1;
	};
};

class Input : public Neuron {
	/**
	 * Particular perceptron which does not change
	 **/
public:
	~Input(){};
	double getResult() {
		return sumPrevious;
	};
};


#endif
