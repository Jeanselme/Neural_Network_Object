#ifndef __ACTIVATION_H
#define __ACTIVATION_H

#include <math.h>

double functionSigmoid(double x);
double derivativeSigmoid(double x);
double functionSigmoidUpdate(double x);
double derivativeSigmoidUpdate(double x);
double functionRectifier(double x);
double derivativeRectifier(double x);

#endif
