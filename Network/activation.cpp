#include "activation.h"

double functionSigmoid(double x) {
	/** Sigmoid function is the most adapted for the backpropagation
	However we can use any differentiable function **/
	return 1. / (1. + exp(-x));
}

double derivativeSigmoid(double x) {
	/** We compute the derivative in order to be faster **/
	return functionSigmoid(x)*(1. - functionSigmoid(x));
}

double functionSigmoidUpdate(double x) {
	return 1.7159*tanh(2./3.*x);
}

double derivativeSigmoidUpdate(double x) {
	return 1.7159*2./3.*(1-pow(tanh(2./3.*x),2));
}


double functionRectifier(double x) {
	return log(1.+exp(x));
}

double derivativeRectifier(double x) {
	return 1./(1.+exp(-x));
}
