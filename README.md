# Neural_Network_Backpropagation
Library of a neural network with backpropagation

## Project organization
The root directory contains an analysis of the MNIST database :  http://yann.lecun.com/exdb/mnist/.

### Extraction
Contains the database and the library for extracting data.

### Network
Contains the network with its nodes and links.
#### Link
This object represents the link between two neurons.
#### Node
This is the neuron.
#### Activation
It is the associated decision function.
#### Network
It is the association of neurons and links in order to have a complete network.

### handwritten_recognition
This file contains a neural network to study the MNIST database.  
Change the different file names of the database for extracting data.

## Neural Network
### Functioning
A neural network is composed of neurons and links between them. In order to compute a value through a neural network, you change the input value and you observe the output of the network.  
However, the network has to be trained on a data set in order to learn the good links (caracterized by a weight) which implies a coherent result.  
This library uses a backpropagation algorithm. It is a chain retropropagation of the gradient, which updates the different weights between nodes by computing the difference between the target and the output values.

### Improvements
In this part, the different improvements that I studied in order to have an efficient neural network.

#### Data Normalization
The extraction puts the input between -0.5 and 0.5.

#### Stochastic gradient
The network computes subset of the training set at each epoch, in order to be faster.

#### Evolution learning rate
Not an adapted solution for stochastic gradient to adapte at each step.  
However, it could be interesting to adapte the learning rate in function of the square root of the number of inputs.  
Moreover, the current algorithm updates the learning rate every (number of element in the training set / number of the epoch) = 100 epochs.

### Results


## Reference
This work is based on the following paper : Efficient BackProp by Yann Lecun, Leon Bottou, Genevieve B. Orr, and Klaus-Robert Müller.
