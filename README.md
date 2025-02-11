# micrograd | A Deep Neural Net from scratch (in Java)

A good first repo to understand the basics of a Deep Neural Net.

## Topics Covered

1. [Introduction](#introduction)
2. [Value class to visualize Mathematical equations](#value-a-class-representing-mathematical-equation)
3. [Structure of a Neural Network](#structure-of-a-neural-network)
      
    1. [Neuron](#neuron)
    2. [Layer](#layer)
    3. [Multi-layer Perceptron](#multilayer-perceptron)

3. [Forward Propagation (Forward Pass)](#forward-propagation)

    1. [Understanding the Mathematics involved](#understanding-the-mathematics-involved-1)
    2. [Activation Functions](#activation-functions)
    3. [Forward Pass in Action](#forward-pass-in-action)

4. [Backpropagation](#backpropagation)

    1. [Understanding the Mathematics involved](#understanding-the-mathematics-involved)

5. [Training Micrograd on the Binary Classifier sample](#training-micrograd-on-the-binary-classifier-sample)
6. [Reference](#reference)
 
## Introduction

**micrograd** is an Autograd engine developed by [Andrej Kerpathy](https://github.com/karpathy). This repo covers the **Java** implementation of [micrograd](https://github.com/karpathy/micrograd). We will start with building a **Value** class which allows us to convey mathematical equations by holding the numerical terms, operators, reference to the parent terms, label and gradient. We will use the Value class to build our **Neuron** over which we will build **Layers** (collection of Neurons) and **Multilayer Perceptron** (collection of Layers) which will lay the foundation of our deep neural network.

![Neural Network](/media/neural-network.png)

We will start with discussing the Value class and how it can convey mathematical equations.

## Value (A class representing mathematical equation)

**Value** class is represented in the following manner.

```
class Value {
  double value;
  Value[] parent;
  double grad;
  String operator;
  String label;
}
```

We can build mathematical equations using the above **Value** object as a building block. Let’s have a sample mathematical equation as shown below:

```y = mx + c```

Let’s build the above equation using the Value class.

```
Value c = new Value(c, "c");
Value m = new Value(m, "m");
Value x = new Value(x, "x");

Value mx = m.multiply(x, "mx");
Value y = mx.add(c, "y");
```

The final Value **y** has the reference to its parents (**mx** and **c**) and so is every Value node in the equation and hence we can backtrack the entire equation. We have shown the above equation (**y = mx + c**) as a graph using Value nodes as building blocks.

![Building Equation via Value class](/media/micrograd1.png)

Since every Value node stores the reference to their parents in the equation, we can backtrack and calculate the gradient of every variable with respect to the final result (in the above equation: y) using the [chain rule](https://en.wikipedia.org/wiki/Chain_rule).

![Chain Rule](/media/chain-rule.png)

We backtrack and use chain rule to **multiply** the gradient of the dependent terms and get the gradient of each term calculated. It is necessary to calculate the gradient of the children's nodes first before computing their parent’ gradient. To ensure this, we perform the [Topological Sorting](https://en.wikipedia.org/wiki/Topological_sorting) of the Mathematical equation graph and compute gradients in the **reverse topological order**.

## Structure of a Neural Network

### Neuron

A Neuron is a building block of our Neural Net and every Neuron object will have some **weights** and **bias**. In order to perform backtracking to compute gradients and preserve the equation, we will keep weights and bias as the **Value** class.

```
class Neuron {
  Value[] weights;
  Value bias;
}
```

A Neuron can be represented as:

![A neuron](/media/neuron-full.png)

Each neuron takes a collection of parameters (x) as an input and assigns weights (w) against each parameter along with a bias (b) added to the result. The equation is as follows:

```
y = w1.x1 + w2.x2 + …. wn.xn + b

```

The result is then given as an input to an **activation function** and the output is considered as the final output of that neuron which might be forwarded to another neuron.

We have used a tanh [activation function](https://paperswithcode.com/method/tanh-activation) in this repo.

```
output = tanh(y)
```

### Layer

Layer is a collection of **Neurons**. It activates each Neuron in their list and returns their output.

A Layer looks like this.

![Layer](/media/layer.jpeg)

### MultiLayer Perceptron

A multi-layer perceptron is a collection of **Layers**. An MLP can have multiple Layers where the output of a layer is an input given to the subsequent layer and so on. The final result of the last layer is considered as an output of the Neural Net. We can configure the **number of layers** and the **number of neurons** present in each layer of our Neural Net architecture.

```
class MultiLayerPerceptron {
  Layer[] layers;
}
```

![MLP](/media/mlp.jpeg)

## Forward Propagation
**Forward Propagation** is the process where the input parameters are passed through the **Layers** present in the **Neural Network** to generate an output. We have explained the Layers in a neural net in the above section (Structure of a Neural Network).

Every Neuron has some **Weights** and **Biases** and they assign their weights to each every input parameter directed to that Neuron. Suppose if a Neuron accepts **5** input parameters then they will assign **5** different weights to those parameters and add a bias to the calculation.

![Forward Pass without activation](/media/forward-pass-1.png)

*Note: The inputs can be the input data which is received by the Neurons present in the **Input Layer** or it can also be the outputs from the Neurons present in the preceding layer which is further accepted by the Neurons present in the current layer.*

### Understanding the Mathematics involved

The output generated by a Neuron is computed via a mathematical equation where the **weights** are assigned to every input parameter and a **bias** is added and then the final result is passed through an **Activation function** to get the output of the Neuron.

Assigning the weights to every input parameter can look somewhat like this:

```
z = (x1w1 + x2w2 + x3w3 + x4w4 + x5w5) + b
```

The resulting output **z** is then passed through an activation function say **f** to get the output of the Neuron.

```
y = f(z)
```

This looks somewhat like this.

![Forward Pass](/media/forward-pass-2.png)

### Activation Functions

The **Activation Function** in an artificial Neural network is a function that calculates the output of the Neuron. It takes the combination of **weights** and **inputs** along with the **bias** as an input and returns the final output.

The input given to an activation function can lie anywhere in the range from **-Infinity** to **+Infinity**. The activation function restricts the range of the output of the neuron to a much smaller window (Eg. from 0 to 1 or from -1 to 1).

Hence, each layer is effectively learning a more complex, higher-level function over the raw inputs and this lets us model very complicated relationships between the inputs and the predicted outputs.

In this repo we have used **Tanh** as an Activation Function which is as follows.

<img src="https://raw.githubusercontent.com/SauravP97/micrograd-java/refs/heads/master/media/tanh.png" width=250>

The growth of the activation function with the inputs looks somewhat like this.

<img src="https://raw.githubusercontent.com/SauravP97/micrograd-java/refs/heads/master/media/activation-function.png" width=400>

We can visualize the tanh activation function that distributes the output of a neuron in a decent range of **[-1 to 1]**.

### Forward Pass in action
We will use the **Value** class to witness the entire Forward Pass procedure in action. The below code snippet performs the Forward Pass in a Neural Network.

```python
w0 = Value(1.3, label="weight: w0")
x0 = Value(1.6, label="input: x0")

w1 = Value(1.2, label="weight: w1")
x1 = Value(0.8, label="input: x1")

w2 = Value(1.1, label="weight: w2")
x2 = Value(0.5, label="input: x2")

b = Value(0.2, label="bias: b")

w0x0 = w0 * x0
w0x0.label = "w0.x0"
w1x1 = w1 * x1
w1x1.label = "w1.x1"
w2x2 = w2 * x2
w2x2.label = "w2.x2"

w0x0w1x1 = w0x0 + w1x1
w0x0w1x1.label = "w0x0 + w1x1"

w0x0w1x1w2x2 = w0x0w1x1 + w2x2
w0x0w1x1w2x2.label = "w0x0 + w1x1 + w2x2"

z = w0x0w1x1w2x2 + b
z.label = "z"

y = z.tanh()
y.label = "y (final output of Neuron)"
```

The above Forward pass equation can be visualized in this way.

![Forward Pass flow](/media/gout_forward_pass.svg)

The Forward Pass happens for all the Neurons in the similar way and the final output is generated by the neurons present in the **Output Layer**.

We further compute the loss by comparing the final output of the Neural net with the actual output and move our way back to reduce the Loss via **Backpropagation** and update the **weights** and **biases** of every neuron using the **gradients** calculated in the backpropagation process.


## Backpropagation

**Backpropagation** is the process of estimating the derivatives or gradients of the parameters present in the **Neural Network** with respect to its **Loss Function** with a goal to **minimize** the Loss value and hence increasing the **accuracy** of the model.

The method uses the concept of **Chain Rule** to determine the gradients of the parameters which are indirectly dependent (through some intermediate parameters) to the Loss Function and influence its value.

The whole idea is to calculate the gradients of the parameters with respect to the loss function and tune the parameters in the opposite direction of the gradient to reduce the loss function. This is a crucial part which happens when the **weights** and **biases** are updated post the backpropagation process.

![Backprop](media/backprop.png)

### Understanding the mathematics involved

Let’s understand the backpropagation process with the help of a simple mathematical equation.

Here we have a mathematical equation to understand the backpropagation process. The equation is inspired from the **activation** process of a Neural Network. Let's take an example of a **Neuron** in a Neural Network fed with **3** input parameters (**x0**, **x1** and **x2**).

Now every Neuron will have **Weights** assigned to these input parameters along with a **Bias** which will be tuned as our model trains and learns the input dataset patterns. The first process is to calculate the output of the Neuron and this process is called **Forward Pass** or **Forward Propagation**.

![A neuron](/media/neuron.jpeg)

The calculation of the output of Neuron happens as follows:

```python
y = w0.x0 + w1.x1 + w2.x2 + b
output = activation_function(y)
```

The result is fed into an **Activation Function** to get the final output of the neuron. We will consider the first part of the above calculation to understand the backpropagation process.

```python
from micrograd.engine import Value

w0 = Value(1.3, label="w0")
x0 = Value(1.6, label="x0")

w1 = Value(1.2, label="w1")
x1 = Value(0.8, label="x1")

w2 = Value(1.1, label="w2")
x2 = Value(0.5, label="x2")

b = Value(0.2, label="b")

w0x0 = w0 * x0
w0x0.label = "w0.x0"
w1x1 = w1 * x1
w1x1.label = "w1.x1"
w2x2 = w2 * x2
w2x2.label = "w2.x2"

w0x0w1x1 = w0x0 + w1x1
w0x0w1x1.label = "w0x0 + w1x1"

w0x0w1x1w2x2 = w0x0w1x1 + w2x2
w0x0w1x1w2x2.label = "w0x0 + w1x1 + w2x2"

y = w0x0w1x1w2x2 + b
y.label = "y"
```

The above code block computes the output y of the neuron. I have used the Value wrapper class to help visualize the equation we built step by step. You can checkout Andrej's [micrograd](https://github.com/karpathy/micrograd) repo to understand more about this.

Let us now visualize the above mathematical equation to understand this in-depth.

![Backprop without grads](/media/backprop-without-grad.png)

The above flowchart depicts the above mathematical equation for calculating y. This graph representation of the equation will further help us in understanding the **Backpropagation** process.

As our next step we will back-propagate though the above equation and calculate the **gradient** / derivative of each terms with respect to the final value **y**.

We will extensively use the [Chain Rule](https://en.wikipedia.org/wiki/Chain_rule) to calculate the gradient of the intermediate terms with respect to the output **y**.

The gradient of all the terms will look like this:

![Backprop with grads](/media/backprop-with-grad.png)

The gradient values in the above diagram shows the influence of each term on the final output y.

In Neural Networks we use the same backpropagation technique on our calculated Loss function. The forward pass provides the Predicted Output from the Neural Net which is then compared with the Actual Output to determine the overall loss. Then we use Backpropagation to find the derivative of the parameters (weights and biases of the neurons) with respect to the Loss function.

We finally adjust the parameters of the neural network with the help of their respective calculated gradients to minimize the overall loss.

This will look somewhat like this.

```python
for parameter in neural_net.parameters():
  parameter.data -= learning_rate * parameter.gradient
```

The entire iteration of Forward and Backward propagation (along with updating the parameters) happens multiple times until the loss gets reduced to an acceptable value.

## Training Micrograd on the Binary Classifier sample

We used a sample dataset of points scattered on a 2D plans. Each dataset row has two parameters X1 and X2 depicting the position of that point on the 2D Plane.
We also have the sample Y which classifies the dataset into red (-1) or blue (1) class.

- Traning data: [train.csv](/dataset/train.csv)
- Predictions made by the neural net: [predictions](/dataset/predictions.txt)

Training micrograd with 2 hidden layers having 16 neurons each on the above sample dataset, classified them into two categories.

<img src="./media/prediction.png" width=300 height=300>

## Reference

- Original repo for micrograd: [Check Out](https://github.com/karpathy/micrograd)
- [Chain Rule](https://en.wikipedia.org/wiki/Chain_rule)
- [Topological Sorting](https://en.wikipedia.org/wiki/Topological_sorting)
- [Activation Function](https://paperswithcode.com/method/tanh-activation)
- [Anrej's github](https://github.com/karpathy)

