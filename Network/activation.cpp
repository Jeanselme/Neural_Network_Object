#include "activation.h"

double functionSigmoid(double x) {
	/** Sigmoid function is the most adapted for the backpropagation
	However we can use any differentiable function **/
	return 1 / (1 + exp(-x));
}

double derivativeSigmoid(double x) {
	/** We compute the derivative in order to be faster **/
	return functionSigmoid(x)*(1 - functionSigmoid(x));
}
