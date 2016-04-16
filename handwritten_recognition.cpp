#include "network.h"
#include "node.h"
#include <stdio.h>

#define NUMBER_LAYER 5
#define INPUT_DIMENSION 28*28

int main( int argc, const char* argv[] )
{
	Network* net = new Network(NUMBER_LAYER);
	
	net->addInputs(INPUT_DIMENSION);
	Bias* b0 = new Bias();
	net->addNode(b0,0);

	net->addNodes(200,1);
	net->fullLinkage(0,1);
	Bias* b1 = new Bias();
	net->addNode(b1,1);

	net->addNodes(100,2);
	net->fullLinkage(1,2);
	Bias* b2 = new Bias();
	net->addNode(b2,2);

	net->addNodes(10,3);
	net->fullLinkage(2,3);
}
