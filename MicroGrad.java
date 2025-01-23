import java.util.*;

/** A Value class representing each numerical nodes or variables. */
class Value {
  double value;
  Value[] parent;
  double grad;
  String operator;
  String label;

  Value(double value, String label) {
    this.value = roundOff(value);
    this.parent = null;
    this.grad = 0.0;
    this.operator = "";
    this.label = label;
  }

  static double roundOff(double value) {
    return (double) Math.round(value * 10000d) / 10000d;
  }

  Value add(Value otherValue, String label) {
    Value returnValue = new Value(this.value + otherValue.value, label);
    returnValue.setParent(new Value[]{this, otherValue});
    returnValue.setOperator("+");

    return returnValue;
  }

  Value subtract(Value otherValue, String label) {
    Value returnValue = new Value(this.value - otherValue.value, label);
    returnValue.setParent(new Value[]{this, otherValue});
    returnValue.setOperator("-");

    return returnValue;
  }

  Value multiply(Value otherValue, String label) {
    Value returnValue = new Value(this.value * otherValue.value, label);
    returnValue.setParent(new Value[]{this, otherValue});
    returnValue.setOperator("*");

    return returnValue;
  }

  Value tanh(String label) {
    double selfValue = this.value;
    double tanhValue = (Math.exp(2.0 * selfValue) - 1.0) / (Math.exp(2.0 * selfValue) + 1.0);
    Value returnValue = new Value(tanhValue, label);
    returnValue.setParent(new Value[]{this});
    returnValue.setOperator("tanh");

    return returnValue;
  }

  ArrayList<Value> getTopologicalOrder() {
    ArrayList<Value> topologicalOrder = new ArrayList<>();
    HashSet<Value> visited = new HashSet<>();

    recurseTopological(this, topologicalOrder, visited);

    // Comment the below lines to print the nodes in topological order.
    // System.out.println("Topological Order: ");
    // for (Value value : topologicalOrder) {
    //   System.out.println(value.label);
    // }

    return topologicalOrder;
  }

  private void recurseTopological(
    Value node, ArrayList<Value> topologicalOrder, HashSet<Value> visited) {
      if (!visited.contains(node)) {
        visited.add(node);
        if (node.parent != null) {
          for (int i=0; i<node.parent.length; i++) {
            Value parentNode = node.parent[i];
            recurseTopological(parentNode, topologicalOrder, visited);
          }
        }
        topologicalOrder.add(node);
      }
  }

  void computeParentGradient() {
    if (this.operator == "+") {
      Value parentNode1 = this.parent[0];
      Value parentNode2 = this.parent[1];

      parentNode1.grad += this.grad * 1.0;
      parentNode2.grad += this.grad * 1.0;
    } else if (this.operator == "*") {
      Value parentNode1 = this.parent[0];
      Value parentNode2 = this.parent[1];

      parentNode1.grad += parentNode2.value * this.grad;
      parentNode2.grad += parentNode1.value * this.grad;
    } else if (this.operator == "tanh") {
      Value parentNode = this.parent[0];
      parentNode.grad += this.grad * (1.0 - Math.pow(this.value, 2.0));
    } else if (this.operator == "-") {
      Value parentNode1 = this.parent[0];
      Value parentNode2 = this.parent[1];

      parentNode1.grad += this.grad * 1.0;
      parentNode2.grad += this.grad * -1.0;
    }
  }

  void setParent(Value[] parent) {
    this.parent = parent;
  }

  void setGrad(double grad) {
    this.grad = grad;
  }

  void setOperator(String operator) {
    this.operator = operator;
  }
}

class Neuron {
  Value[] weights;
  Value bias;

  Neuron(int inputs) {
    weights = new Value[inputs];
    for (int i=0; i<inputs; i++) {
      weights[i] = new Value(Math.random(), "w"+Integer.toString(i+1));
    }
    bias = new Value(Math.random(), "b");
  }

  Value activate(Value[] x) {
    Value activatedValue = new Value(0.0, "output");

    for (int i = 0; i < x.length; i++) {
      activatedValue = activatedValue.add(
        weights[i].multiply(
          new Value(x[i].value, "x" + Integer.toString(i+1)), 
          "x" + Integer.toString(i+1) + "w" + Integer.toString(i+1)), "z");
    }
    
    activatedValue = activatedValue.add(bias, "z");

    return activatedValue.tanh("tan(z)");
  }

  List<Value> parameters() {
    List<Value> parameters = new ArrayList<>();
    for (int i=0; i<weights.length; i++) {
      parameters.add(weights[i]);
    }
    parameters.add(bias);

    return parameters;
  }
}

class Layer {
  Neuron[] neurons;

  Layer(int neuron_count, int inputs) {
    neurons = new Neuron[neuron_count];
    for (int i=0; i<neuron_count; i++) {
      neurons[i] = new Neuron(inputs);
    }
  }

  Value[] activate(Value[] x) {
    Value[] outputs = new Value[neurons.length];
    int index = 0;
    for (Neuron neuron : neurons) {
      outputs[index] = neuron.activate(x);
      index++;
    }

    return outputs;
  }
  
  List<Value> parameters() {
    List<Value> parameters = new ArrayList<>();

    for (int i=0; i<neurons.length; i++) {
      parameters.addAll(neurons[i].parameters());
    }

    return parameters;
  }
}


class MultiLayerPreceptron {
  Layer[] layers;

  // Eg. layerDistribution = [4, 4, 1]
  // It means the MLP has 3 layers each having 4, 4 and 1 neurons.
  MultiLayerPreceptron(int inputs, int[] layerDistribution) {
    layers = new Layer[layerDistribution.length];
    layers[0] = new Layer(layerDistribution[0], inputs);

    for (int i=1; i<layerDistribution.length; i++) {
      layers[i] = new Layer(layerDistribution[i], layerDistribution[i-1]);
    }
  }

  Value[] activate(Value[] x) {
    for (Layer layer : layers) {
      x = layer.activate(x);
    }

    return x;
  }

  List<Value> parameters() {
    List<Value> parameters = new ArrayList<>();

    for (int i=0; i<layers.length; i++) {
      parameters.addAll(layers[i].parameters());
    }

    return parameters;
  }
}

public class MicroGrad {
  private static Value[][] buildTrainingData() {
    Value[][] train = {
      new Value[]{new Value(2.0, "x1"), new Value(3.0, "x2"), new Value(-1.0, "x3")},
      new Value[]{new Value(3.0, "x1"), new Value(-1.0, "x2"), new Value(0.5, "x3")},
      new Value[]{new Value(0.5, "x1"), new Value(1.0, "x2"), new Value(1.0, "x3")},
      new Value[]{new Value(1.0, "x1"), new Value(1.0, "x2"), new Value(-1.0, "x3")}
    };

    return train;
  }

  private static Value[] buildLabelOutput() {
    return new Value[]{new Value(1.0, "y1"), new Value(-1.0, "y2"), new Value(-1.0, "y3"), new Value(1.0, "y4")};
  }

  public static void main(String[] args) {
    // Initialize Neural Net
    MultiLayerPreceptron mlp = new MultiLayerPreceptron(3, new int[]{4, 4, 1});

    // Forward Propogation
    // Value[] test1 = new Value[]{new Value(2.0, "x1"), new Value(3.0, "x2"), new Value(-1.0, "x3")};
    // Value[] test2 = new Value[]{new Value(3.0, "x1"), new Value(-1.0, "x2"), new Value(0.5, "x3")};
    // Value[] test3 = new Value[]{new Value(0.5, "x1"), new Value(1.0, "x2"), new Value(1.0, "x3")};
    // Value[] test4 = new Value[]{new Value(1.0, "x1"), new Value(1.0, "x2"), new Value(-1.0, "x3")};

    Value[][] train = buildTrainingData();
    Value[] actual = buildLabelOutput();
    double learningRate = 0.01;

    // Iterations
    for (int x = 0; x<3; x++) {
      Value[][] pred = new Value[actual.length][1];
      Value netLoss = new Value(0.0, "netLoss");

      for (int i=0; i<train.length; i++) {
        pred[i] = mlp.activate(train[i]);
        Value curLoss1 = pred[i][0].subtract(actual[i], "loss1");
        Value curLoss2 = pred[i][0].subtract(actual[i], "loss2");
        Value curLoss = curLoss1.multiply(curLoss2, "loss");

        netLoss = netLoss.add(curLoss, "netLoss");
      }
    
      System.out.println("Net Loss: ");
      printNode(netLoss, null);

      ArrayList<Value> topologicalOrder = netLoss.getTopologicalOrder();
    
      // Traversing in the reverse Topological order to make sure the
      // dependencies have their gradient calculated beforehand.
      Collections.reverse(topologicalOrder);

      // Back Propogation
      netLoss.grad = 1.0;
      for (Value value : topologicalOrder) {
        value.computeParentGradient();
      }

      // Print Nodes
      // printNode(netLoss, null);
      // traverseToTop(netLoss);

      // Adjust the parameters of the Neural Net post Back prop.
      List<Value> parameters = mlp.parameters();
      System.out.println(parameters.size());
      for (Value parameter : parameters) {
        parameter.value -= learningRate * parameter.grad;
      }
    }
  }

  private static void traverseToTop(Value node) {
    if (node.parent == null || node.parent.length == 0) {
        return;
    }

    for (int i=0; i<node.parent.length; i++) {
        Value parentNode = node.parent[i];
        printNode(parentNode, node);
        traverseToTop(parentNode);
    }
  }

  private static void printNode(Value node, Value childNode) {
    String child = childNode != null ? childNode.label : "--";
    System.out.println("==== Node ====");
    System.out.println("Value: " + node.value);
    System.out.println("Operator: " + node.operator);
    System.out.println("Child: " + child);
    System.out.println("Label: " + node.label);
    System.out.println("Derivative with respect to L: " + node.grad);
    System.out.println();
  }
}