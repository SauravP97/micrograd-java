# micrograd | A Neural Net from scratch (in Java)

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

## Neuron

A Neuron is a building block of our Neural Net and every Neuron object will have some **weights** and **bias**. In order to perform backtracking to compute gradients and preserve the equation, we will keep weights and bias as the **Value** class.

```
class Neuron {
  Value[] weights;
  Value bias;
}
```

A Neuron can be represented as:

![A neuron](/media/neuron.jpeg)

Each neuron takes a collection of parameters (x) as an input and assigns weights (w) against each parameter along with a bias (b) added to the result. The equation is as follows:

```
y = w1.x1 + w2.x2 + …. wn.xn + b

```

The result is then given as an input to an **activation function** and the output is considered as the final output of that neuron which might be forwarded to another neuron.

We have used a tanh [activation function](https://paperswithcode.com/method/tanh-activation) in this repo.

```
output = tanh(y)
```

## Layer

Layer is a collection of **Neurons**. It activates each Neuron in their list and returns their output.

A Layer looks like this.

![Layer](/media/layer.jpeg)

## MultiLayer Perceptron

A multi-layer perceptron is a collection of **Layers**. An MLP can have multiple Layers where the output of a layer is an input given to the subsequent layer and so on. The final result of the last layer is considered as an output of the Neural Net. We can configure the **number of layers** and the **number of neurons** present in each layer of our Neural Net architecture.

```
class Neuron {
  Layer[] layers;
}
```

![MLP](/media/mlp.jpeg)

## Training Micrograd on the Binary Classifier sample

We used a sample dataset of points scattered on a 2D plans. Each dataset row has two parameters X1 and X2 depicting the position of that point on the 2D Plane.
We also have the sample Y which classifies the dataset into red (-1) or blue (1) class.

Traning data: [train.csv](/dataset/train.csv)
Predictions made by the neural net: [predictions](/dataset/predictions.txt)

Training micrograd with 2 hidden layers having 16 neurons each on the above sample dataset, classified them into two categories.

![Prediction](/media/prediction.png)

## Reference

- Original repo for micrograd: [Check Out](https://github.com/karpathy/micrograd)

