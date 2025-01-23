import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;

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
