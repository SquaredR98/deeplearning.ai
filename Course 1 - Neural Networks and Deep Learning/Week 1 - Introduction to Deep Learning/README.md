# Introduction to Deep Learning

## What is a Neural Network?

- **Single Neuron**: Linear regression without applying activation(perceptron) which looks likes as follows:

  ![](./images/Perceptron.png)

- A single neuron calculates weighted sum of inputs i.e. <a href="https://www.codecogs.com/eqnedit.php?latex=\textbf&space;W^{T}\textbf&space;X" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\textbf&space;W^{T}\textbf&space;X" title="\textbf W^{T}\textbf X" /></a> and then we set a threshold to predict output in perceptron.

- A perceptron can take real valued input or boolean valued and when <a href="https://www.codecogs.com/eqnedit.php?latex=W^{T}&space;X&space;&plus;&space;b&space;=&space;0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?W^{T}&space;X&space;&plus;&space;b&space;=&space;0" title="W^{T} X + b = 0" /></a> then the perceptron outputs 0.

- **Disadvantage of Perceptron**: Perceptron only outputs binary values and if we try to give small change in weight and bias then perceptron can flip the output.

- To overcome this we use the sigmoid function which makes slight changes to the output with the changes in weights and biases. For example, If the output using some weights and bias in perceptron is 0 and we change the weights and bias slightly then the output becomes = 1 but the actual output in that case was 0.7. In case of sigmoid, initial output = 0 but with slight modification output will be exact same 0.7.

- Applying sigmoid activation function to a single neuron makes it to act like logistic regression.

- The difference between sigmoid and perceptron can be visualized as follows:

  ![](./images/sigmoidvsperceptron.png)

- Simple Neural Network:

  <img src="./images/nnsimple.png" height="300" style="text-align:center"/>

- **ReLu**: It stands for Rectified Linear Unit. It is most popular activation function making deep neural network training faster.

- Hidden layer predicts connection between inputs automatically, that's what deep learning is good at.

- Deep Neural Network consists of more hidden layers (Deeper Layers): 

  ![](./images/deepnn.png)

- Each input in the neural network is connected to the hidden layer and the neural network decides all the connections.

## Supervised Learning with Neural Network
