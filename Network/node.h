#ifndef __NODE_H
#define __NODE_H

#include "activation.h"
#include <assert.h>
#include <string.h>
#include <omp.h>
#include <vector>

using namespace std;

class Link;

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

	// Axion input
	vector<Link*> previous;

public:
	Neuron() {
		memset(sumPrevious, 0, OMP_NUM_THREADS);
		memset(delta, 0, OMP_NUM_THREADS);
		memset(computed, false, OMP_NUM_THREADS);
		memset(deltaComputed, false, OMP_NUM_THREADS);
	};

	virtual ~Neuron() {};

	void addPrevious(Link* link) {
		previous.push_back(link);
	};

	vector<Link*> getPrevious() {
		return previous;
	}

	virtual double getResult(int tid = 0) {
		if (!computed[tid]) {
			#pragma omp critical
			{
				result[tid] = functionSigmoid(sumPrevious[tid]);
				computed[tid] = true;
			}
		}
		return result[tid];
	};

	double getDelta(int tid = 0) {
		if (!deltaComputed[tid]) {
			deltaResult[tid] = derivativeSigmoid(result[tid])*delta[tid];
			deltaComputed[tid] = true;
		}
		return deltaResult[tid];
	};

	void addDelta(double delta_to_add, int tid = 0) {
		delta[tid] += delta_to_add;
	};

	void addSum(double val_to_add, int tid = 0) {
		sumPrevious[tid] += val_to_add;
	};

	void reinitDelta(int tid = 0) {
		delta[tid] = 0;
		deltaComputed[tid] = false;
	};

	void reinitSum(int tid = 0) {
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
	double getResult(int tid = 0) {
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
	double getResult(int tid = 0) {
		return sumPrevious[tid];
	};
};


#endif
