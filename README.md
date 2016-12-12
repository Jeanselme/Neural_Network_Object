# Neural_Network_Object
Library of a neural network with backpropagation

## Why Object ?
My first idea was to create a neural network which would be easy to manipulate. And to allow to create evolutive neural network. Each node is an object which is easy to change, destroy or add.  
However, that implies lower performances than matricial neural network.

## Project organization
The root directory contains an analysis of the MNIST database :  http://yann.lecun.com/exdb/mnist/.

### Extraction
Contains library for extracting data.

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

#### Regularization
The weight is added to the loss function, in order to reduce the l2-norm of the weight.

#### Parallelization
The learning phase is parallelized on the data, ie each thread computes a certain number of images of a batch. So the number of thread has to be less then the number of images by batch.  

The following graphs show the learning execution time for different neural networks, with a batch of 100 images.  
Tests are realised on a Intel i7-4750HQ @ 2.00GHz x 8 (only 4 real) with 16Gio of RAM.  

For 50 hidden nodes, and 10 iterations :  
![Result](https://raw.githubusercontent.com/Jeanselme/Neural_Network_Object/master/Images/50-10-100.png)  
The cost of parallelization is interesting for more than 2 threads.  

For 500 hidden nodes and 1 iteration :  
![Result](https://raw.githubusercontent.com/Jeanselme/Neural_Network_Object/master/Images/500-1-100.png)  
This configuration seems curious, it is certainly due to my cpu architecture.  

For 5000 hidden nodes and 100 iteration :  
![Result](https://raw.githubusercontent.com/Jeanselme/Neural_Network_Object/master/Images/5000-10-100.png)  
This shows the interest of parallelize the neural network.  
However it would be interesting to study it with larger batch and more numerous threads.

## How to launch it ?
```
make download
```
In order to download and extracts data. If you have not python3.5 and gunzip, please download the data on Lecun website and unzip the different database in Data folder.  

```
make run
```
To run the handwritten example with a simple neural network.  

## Future enhancements
- Parallelize the computation phase in order to speed up the computation.
- Adapte the learning rate thanks to its number of incoming links. It would be interesting to see the impact of an evoluting regularization rate, which incresases when the neural network has good results.  
- Create evolutive neural network
- Create convolutional one.

## Reference
This work is based on the following paper : Efficient BackProp by Yann Lecun, Leon Bottou, Genevieve B. Orr, and Klaus-Robert Müller.
