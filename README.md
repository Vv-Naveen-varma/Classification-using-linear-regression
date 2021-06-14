# Classification using Linear regression model using neural networks

### We are going to build a Linear classification model to understand the neural network concepts 
> How do we build a model? 

> How do we know that our model is working properly?

All these questions will be answered.

### First let's start with a linear model example to classify whether a person has diabetes or not

## Target

![image1](https://user-images.githubusercontent.com/63995834/121881935-84ca3980-cd2d-11eb-90f5-b4d14e139547.jpg)

Graph Description:
* Each point in the graph represents a person
* Points in blue color represents preson who does not have diabetes
* points in red color represents person who has diabetes

* As shown above, considering age and boold gluocse levels of a person on x and y axis, we need to find a line that best classifies the data. So that, when a new person's details are entered our model must be able to classify which group the person belongs to. In other words, whether the person has a risk of diabetes or not.

final linear model looks like this:

![image2](https://user-images.githubusercontent.com/63995834/121881966-8dbb0b00-cd2d-11eb-8a70-1cf873641e30.jpg)

## Model building

* A perceptron is a basic form of Neural network that takes inspiration from the brain. What does a brain do? takes input from our ears, eyes, nose, process it and produce a result. Perceptron is pretty similar.

![image3](https://user-images.githubusercontent.com/63995834/121881999-97dd0980-cd2d-11eb-9efa-317241d97f4c.jpg)

* Step-1: lets consider an example, place a random linear model into a node, we can call it a "model node or perceptron". As we discussed early, like brain, our model node is going to receives inputs age(x1) and bloog glucose(x2). Take a random equation or line 

            A line can be represented as 

                w1x1 + w2x2 + b = 0

        where w1, w2 are weights which dictates slope and b is the bias

* Step-2. These weights start out as random values. So at the beginning, we are just going to have a random line which does not classify our data correctly.
* Step-3. But as the neural network learns more about what kind of output data its dealing with, it will adjust the weights based on the output errors that resulted in categorizing the data with previous weights, until it comes up with a better model. So, how do we do this?

![image4](https://user-images.githubusercontent.com/63995834/121882012-9dd2ea80-cd2d-11eb-8b05-f0c1655144b0.jpg)

* Step-4: Considering our selected random model, we use Sigmoid function to predict continuous probabilities for each point. This function is also known as activation function. What is this Sigmoid function?

### Sigmoid Function Theory:

* We discussed that the system starts with a random linear model to separate our data then calculates the errors associated with this model and then readjust the weights to minimize the error and properly classify the data points. Now comes the question How to calculate the error? We be needing a continuous error function.

![image10](https://user-images.githubusercontent.com/63995834/121882032-a4616200-cd2d-11eb-8715-76f39e808f8f.jpg)

* Looking at the above diagram clearly there are two misclassified points. We know the blue points need to be below the line and red one's above the line.
* So, the error function is going to assign each misclassified point a big penalty. For better understanding, we set the size of the points reflect the size of penalty in the below image. What we do is detect these error variations and thus figure out which direction we need to move the line the most. The total error results from the sum of these penalties associated with each point.

![image11](https://user-images.githubusercontent.com/63995834/121882053-a9261600-cd2d-11eb-91c4-e309191ab22f.jpg)

* From the image above, we can see that there is high error value. So we move the line in the direction of the most errors, as shown below, until all error penalties are sufficiently small, thus minimizing the error.

![image12](https://user-images.githubusercontent.com/63995834/121882072-afb48d80-cd2d-11eb-9455-37761194bb7a.jpg)                   ![image13](https://user-images.githubusercontent.com/63995834/121882099-b6db9b80-cd2d-11eb-89a9-e221dc5f7d65.jpg)                   ![image14](https://user-images.githubusercontent.com/63995834/121882116-bb07b900-cd2d-11eb-8395-d6a74a85d7b4.jpg)                   ![image15](https://user-images.githubusercontent.com/63995834/121882128-bf33d680-cd2d-11eb-81bd-960fbc351f29.jpg)

* Lets re-think our perceptron model

![image16](https://user-images.githubusercontent.com/63995834/121882141-c4912100-cd2d-11eb-9147-f5798099124e.jpg)

* In the second node, based on the score of each point it predicts a value of 0 or 1. Any point with positive score gets a 1 otherwise 0. These are discrete predictions which are derived from our step function. the problem is a step function incereases or decreases very abruptly from one constant value to another. There is no inbetween.
* We need continuous probabilities that is why we cannot use step function, as step function just tells us yes or no. So, we use sigmoid funciton instead of step function.

image17![image17](https://user-images.githubusercontent.com/63995834/121882152-ca870200-cd2d-11eb-9f50-7fdbb40e4b38.jpg)

* Step-5: Using these probabilities we calculate the error with Cross Entropy. So what is Cross entropy? 

### Cross Entropy Theory:

            It is an error function used to calculate the total error associated with our linear model, Remember "more incorrect our model in separating our data-more the entropy value", thus larger the error.

* The idea is that with some random displayed data the computer will display some random model, based on that model it needs to calculate the error.

* If you look at the example given below, you can see how a cross entropy is calculated for a point. Here y = label where label = 1 as the blue point indicates that a person has diabetes, probability p = 0.95, using the formula calculate probability of a point being blue.

![image5](https://user-images.githubusercontent.com/63995834/121882162-cf4bb600-cd2d-11eb-9b2f-cd3469480c27.jpg)

* In the same way we calculate the probability of each and every point to find the total error associated with our model.

![image6](https://user-images.githubusercontent.com/63995834/121882174-d377d380-cd2d-11eb-8050-6dc0cd7d8654.jpg)

* Let us consider two models good model and a bad model, as shown below, and calculate total error for both models to observe the difference. If you look at the below example models, you can easily identify which is the best model, the model that has less entropy value (error value) is the model that can classify more accurately, that is the one on the right.

![image7](https://user-images.githubusercontent.com/63995834/121882189-d96db480-cd2d-11eb-9ca8-7a50ab45c95f.jpg)

* Step-6: Using these cross entropy values we apply gradient descent, which keeps minimizing the error, doing so obtaining the linear model that better classifies.

So, what is this Gradient Descent?

### Gradient Descent Theory:
* Previously we looked at calculating the total error with our cross entropy function. But now what we'll do is use gradient descent to minimze that error and obtain a better model which better classifies the data and we keep doing that over and over through many iterations until we obtain the perfect line.
* To minimze the erro, we need to take its gradient (It is the derivative with respect to weights). If we subtract the gradient to our linear parameters [weight1, weight2 and bias] it tells us the value thats going to decrease the error function the most. Ultimately resulting in a linear model with a smaller error.
* This process is repeated, thus minimizing the error and evetually obtaining a line with small enough error that correctly classifies our data.

![image8](https://user-images.githubusercontent.com/63995834/121882206-dffc2c00-cd2d-11eb-878a-df5bfe9a50a2.jpg)

* Unfortunatley in Python we're not able to simply take the derivative of the error function but have to actually derive the equation ourselves and then code it.

![image9](https://user-images.githubusercontent.com/63995834/121882961-d1624480-cd2e-11eb-83b3-bdf6328a2a1d.jpg)

* Step-7: Finally, by following all these steps I implemented a linear classification model using neural network.

## Note
These are the required basics for understanding the concepts of deep neural network. Now that you understood these, you can proceed to Deep Neural Networks models click on

                        Link
