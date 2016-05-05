#ifndef __EXTRACTION_H
#define __EXTRACTION_H

#include <iostream>
#include <fstream>
#include <vector>
 
using namespace std;

int reverseInt (int i);
int readMNIST(const char* database, const char* labelname,vector< vector<double> > &images, vector< vector<int> > &labels);

#endif