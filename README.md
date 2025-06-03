Basic feed-forward neural network (FNN) made entirely with NumPy, trained for handwritten digits recognition.

Architecture:
- 1 input layer with 784 neurons (one per image pixel) and ReLU activation function
- 1 hidden layer with 10 neurons and ReLU activation function
- 1 output layer with 10 neurons and softmax activation function (probabilities for each digit)

We've used cross-entropy cost function.

The project is inspired by [this](http://neuralnetworksanddeeplearning.com/chap2.html) article, which has great 
introduction to neural networks with all backpropagation equations derived. We're also aware of very similar project
[here](https://www.youtube.com/watch?v=w8yWXqWQYmU&t=551s),
although we did not use it as a reference.

## Results
We've trained the model for 20 epochs with `batch_size=32` and `learning_rate=0.001`. Train and test datasets have
60 000 and 10 000 images respectively. We've achieved 84 % accuracy (learned parameters are saved and included in the project).

This shows that FNN is implemented correctly and is capable of learning, which accomplishes the main goal of illustrating 
basic deep learning concepts. 

However, the achieved accuracy is far frome optimal and can't be significantly improved by further learning or addition of hidden layers
(we were never able to break 90 % benchmark). The same results could be achieved using, for example, decision tree with way less computational effort.

Modern neural networks have additional features like learning rate schedulers and advanced optimizers, which greatly facilitate learning process.

Additionally, it is well-known that convolutional neural networks perform way better in image recognition tasks. 
