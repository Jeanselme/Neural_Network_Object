#ifndef __RANDOM_H
#define __RANDOM_H

#include <stdlib.h>
#include <time.h>

class Random {
	/**
	 * Link between two neurons
	 **/
private:
 static Random *instance;
 Random() {
	 srand((int)time(NULL));
 };

public:
	static Random *get() {
		if (!instance) {
			instance = new Random();
		}
		return instance;
	}

	double getRandom() {
		return ((static_cast <float> (rand())
			/ static_cast <float> (RAND_MAX)));
	};
};

#endif
