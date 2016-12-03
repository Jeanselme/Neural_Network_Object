#ifndef __NODE_H
#define __NODE_H

#include "activation.h"
#include <assert.h>
#include <string.h>
#include <omp.h>

class Neuron {
	/**
	 * Class for representing a perceptron
	 **/
protected:
	// Sum which will be computed in order
	// to obtain the output of the neuron
	double sumPrevious[OMP_NUM_THREADS];
	// Delta of the neuron
	double delta[OMP_NUM_THREADS];

	// To avoid computation repetition
	double result[OMP_NUM_THREADS];
	bool computed[OMP_NUM_THREADS];

	double deltaResult[OMP_NUM_THREADS];
	bool deltaComputed[OMP_NUM_THREADS];

public:
	Neuron() {
		memset(sumPrevious, 0, OMP_NUM_THREADS);
		memset(delta, 0, OMP_NUM_THREADS);
		memset(computed, false, OMP_NUM_THREADS);
		memset(deltaComputed, false, OMP_NUM_THREADS);
	};

	virtual ~Neuron() {};

	virtual double getResult(int tid = 0) {
		if (!computed[tid]) {
			result[tid] = functionSigmoid(sumPrevious[tid]);
			computed[tid] = true;
		}
		return result[tid];
	};

	double getDelta(int tid) {
		if (!deltaComputed[tid]) {
			deltaResult[tid] = derivativeSigmoid(result[tid])*delta[tid];
			deltaComputed[tid] = true;
		}
		return deltaResult[tid];
	};

	void addDelta(double delta_to_add, int tid) {
		delta[tid] += delta_to_add;
	};

	void addSum(double val_to_add, int tid) {
		sumPrevious[tid] += val_to_add;
	};

	void reinitDelta(int tid) {
		delta[tid] = 0;
		deltaComputed[tid] = false;
	};

	void reinitSum(int tid) {
		sumPrevious[tid] = 0;
		computed[tid] = false;
	};

};

class Bias : public Neuron {
	/**
	 * Particular perceptron which does not compute
	 **/
public:
	~Bias() {};
	double getResult(int tid) {
		assert(tid < OMP_NUM_THREADS);
		return 1;
	};
};

class Input : public Neuron {
	/**
	 * Particular perceptron which does not change
	 **/
public:
	~Input(){};
	double getResult(int tid) {
		return sumPrevious[tid];
	};
};


#endif
