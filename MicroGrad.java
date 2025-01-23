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

    System.out.println("Topological Order: ");
    for (Value value : topologicalOrder) {
      System.out.println(value.label);
    }

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
      weights[i] = new Value(Math.random(), "w"+Integer.toString(i));
    }
    bias = new Value(Math.random(), "b");
  }

  double activate(double[] x) {
    Value activatedValue = new Value(0.0, "output");

    for (int i = 0; i < x.length; i++) {
      activatedValue = activatedValue.add(
        weights[i].multiply(
          new Value(x[i], "x" + Integer.toString(i)), 
          "x" + Integer.toString(i) + "w" + Integer.toString(i)), "z");
    }
    
    activatedValue = activatedValue.add(bias, "z");

    return activatedValue.tanh("tan(z)").value;
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

  double[] activate(double[] x) {
    double[] outputs = new double[neurons.length];
    int index = 0;
    for (Neuron neuron : neurons) {
      outputs[index] = neuron.activate(x);
      index++;
    }

    return outputs;
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

  double[] activate(double[] x) {
    for (Layer layer : layers) {
      x = layer.activate(x);
    }

    return x;
  }
}

public class MicroGrad {
  public static void main(String[] args) {
    // neuron();
    MultiLayerPreceptron mlp = new MultiLayerPreceptron(3, new int[]{4, 4, 1});
    double[] outputs = mlp.activate(new double[]{2.0, 3.0, -1.0});
    System.out.println(Arrays.toString(outputs));
  }

  private static void neuron1() {
    // Equation: n = x1.w1 + x2.w2 + b
    // Equation: o = tanh(n)

    // Inputs x1, x2
    Value x1 = new Value(2.0, "x1");
    Value x2 = new Value(0.0, "x2");
    
    // Weights w1, w2
    Value w1 = new Value(-3.0, "w1");
    Value w2 = new Value(1.0, "w2");

    // Bias of the neuron
    Value b = new Value(6.88, "b");

    Value x1w1 = x1.multiply(w1, "x1.w1");
    Value x2w2 = x2.multiply(w2, "x2.w2");
    Value x1w1x2w2 = x1w1.add(x2w2, "x1.w1 + x2.w2");

    Value n = x1w1x2w2.add(b, "n");
    Value o = n.tanh("o");

    ArrayList<Value> topologicalOrder = o.getTopologicalOrder();
    
    // Traversing in the reverse Topological order to make sure the
    // dependencies have their gradient calculated beforehand.
    Collections.reverse(topologicalOrder);

    o.grad = 1.0;
    for (Value value : topologicalOrder) {
      value.computeParentGradient();
    }
    
    printNode(o, null);
    traverseToTop(o);
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