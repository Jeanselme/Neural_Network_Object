#ifndef __RANDOM_H
#define __RANDOM_H

#include <stdlib.h>
#include <string.h>
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

	~Random() {
		free(instance);
	}

	double getRandom() {
		return ((static_cast <float> (rand())
			/ static_cast <float> (RAND_MAX)));
	};

	/**
		* Returns an array of the given size with half true and half false
		**/
	void randomBoolean(int length, bool* array, double percentage) {
		memset(array, false, length);
		for (int i = 0; i < length*percentage; i++) {
			bool ok = false;
			// Reject in order to have exactly half at each iteration
			// TODO : Find a cheaper way to do that
			while (!ok) {
				int indice = (int) (getRandom() * length);
				if (!array[indice]) {
					array[indice] = true;
					ok = true;
				}
			}
		}
	}
};

#endif
