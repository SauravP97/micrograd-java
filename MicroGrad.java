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

  void computeParentGradient() {
    if (this.operator == "+") {
        Value parentNode1 = this.parent[0];
        Value parentNode2 = this.parent[1];

        parentNode1.grad = this.grad * 1.0;
        parentNode2.grad = this.grad * 1.0;
    } else if (this.operator == "*") {
        Value parentNode1 = this.parent[0];
        Value parentNode2 = this.parent[1];

        parentNode1.grad = parentNode2.value * this.grad;
        parentNode2.grad = parentNode1.value * this.grad;
    } else if (this.operator == "tanh") {
        Value parentNode = this.parent[0];
        parentNode.grad = this.grad * (1.0 - Math.pow(this.value, 2.0));
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

public class MicroGrad {
  public static void main(String[] args) {
    double h = 0.0001;
    double x = 2.0/3.0;
    double derivative = (f(h + x) - f(x)) / h;

    neuron();
  }

  private static void neuron() {
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

    // do/do = 1
    o.grad = 1.0;
    o.computeParentGradient();
    n.computeParentGradient();
    x1w1x2w2.computeParentGradient();
    x1w1.computeParentGradient();
    x2w2.computeParentGradient();

    // // do/dn = 1 - tanh(x)^2
    // n.grad = 1 - o.multiply(o, "dn").value;

    // // do/dx1w1x2w2 = do/dn * dn/dx1w1x2w2
    // x1w1x2w2.grad = n.grad;

    // // do/db = do/dn * dn/db
    // b.grad = n.grad;

    // // do/dx1dw1 = do/dx1w1x2w2 * dx1w1x2w2/dx1w1
    // x1w1.grad = x1w1x2w2.grad;

    // // do/dx2dw2 = do/dx1w1x2w2 * dx1w1x2w2/dx2w2
    // x2w2.grad = x1w1x2w2.grad;

    // // do/dx1 = do/dx1w1 * dx1w1/dx1
    // x1.grad = x1w1.grad * w1.value;

    // // do/dx2 = do/dx2w2 * dx2w2/dx2
    // x2.grad = x2w2.grad * w2.value;

    // // do/dw1 = do/dx1w1 * dx1w1/dw1
    // w1.grad = x1w1.grad * x1.value;

    // // do/dw2 = do/dx2w2 * dx2w2/dw2
    // w2.grad = x2w2.grad * x2.value;
    
    printNode(o, null);
    traverseToTop(o);
  }

  private static double lossValue() {
    Value h = new Value(0.001, "h");

    // Inputs
    double a = 2.0;
    double b = -3.0;
    double c = 10.0;
    double f = -2.0;

    Value l = computeFinalResult(a, b, c, f);

    printNode(l, null);
    traverseToTop(l);

    return l.value;
  }

  private static Value computeFinalResult(double av, double bv, double cv, double fv) {
    // Equation: 
    // L = f * ((a * b) + c)
    // L = f * (e + c)
    // L = f * d
    Value a = new Value(av, "a");
    Value b = new Value(bv, "b");
    Value c = new Value(cv, "c");
    Value e = a.multiply(b, "e");
    Value d = e.add(c, "d");
    Value f = new Value(fv, "f");
    Value l = d.multiply(f, "L");

    l.grad = 1;
    f.grad = d.value;
    d.grad = f.value;

    // dL/de = dL/dd * dd/de
    e.grad = f.value * 1.0;

    // dL/dc = dL/dd * dd/dc
    c.grad = f.value * 1.0;

    // dL/da = dL/de * de/da
    a.grad = e.grad * b.value;

    // dL/db = dL/de * de/db
    b.grad = e.grad * a.value;

    return l;
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

  private static double f(double x) {
    return (3.0 * Math.pow(x, 2.0)) - (4.0 * x) + 5.0;
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