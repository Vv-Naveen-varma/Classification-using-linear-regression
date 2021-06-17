# Classification with Linear regression model using neural networks

### Have a look at the concepts of neural networks at [medium](https://naveen-varma.medium.com/linear-classification-model-using-neural-networks-basics-of-deep-neural-networks-2f37fa8f07bb)

> Note that all neural networks are referred as Artificial Neural Networks(ANN). The neural networks with more than one hidden layer are called Deep Neural Networks(DNN). Convolutional Neural Networks(CNN) are mainly used in image processing.

### We are going to build a Linear binary classification model to understand the neural network concepts

## Theory:
    A line can be represented as 
    
    w1x1 + w2x2 + b = 0 
    
    where w1, w2 are weights which dictate a slope and b is the bias

* These weights start out as random values. So, we are just going to have a random line which does not classify our data correctly.
* But as the neural network learns more about what kind of output data its dealing with it will adjust the weights based on the output errors that resulted in categorizing the data with previous weights, until it comes up with a better model.So, how do we do this?
* We use Sigmoid function to predict continuos probabilities for each point.
* Using these probabilities we calculate the error with Cross Entropy [It is an error function used to calculate the total error associated with our linear model, the more incorrect our model in separating our data more the entropy value, thus larger the error]
* We use gradient descent, which keeps minimizing the error, doing so obtaining the linear model.
