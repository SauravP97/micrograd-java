import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.Queue;

public class MicroGrad {
  public static void main(String[] args) {
    // Initialize Neural Net
    MultiLayerPreceptron mlp = new MultiLayerPreceptron(3, new int[]{4, 4, 1});

    Value[][] train = buildTrainingData();
    Value[] actual = buildLabelOutput();
    double learningRate = -0.05;

    // Iterations
    for (int x = 0; x<100; x++) {
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

      // Flush out the gradient values (zero_grad)
      List<Value> parameters = mlp.parameters();
      for (Value parameter : parameters) {
        parameter.grad = 0;
      }

      // Back Propogation
      netLoss.grad = 1.0;
      for (Value value : topologicalOrder) {
        value.computeParentGradient();
      }

      // printNode(netLoss, null);
      // traverseToTop(netLoss, 0);

      // Adjust the parameters of the Neural Net post Back prop.
      for (Value parameter : parameters) {
        System.out.println("Gradient: " + parameter.grad + " " +  parameter.label);
        parameter.value += learningRate * parameter.grad;
      }

      System.out.println("Predicted Values: " + pred[0][0].value + " " + pred[1][0].value + " " + pred[2][0].value + " " + pred[3][0].value);
    }
  }

  private static void traverseToTop(Value node, int count) {
    System.out.println("Counter: " + count);
    if (node.parent == null || node.parent.length == 0) {
        return;
    }

    Queue<Value> queue = new LinkedList<>();
    queue.add(node);

    while (!queue.isEmpty()) {
      Value childNode = queue.remove();
      if (childNode.parent == null || childNode.parent.length == 0) {
        continue;
      }

      for (int i=0; i<childNode.parent.length; i++) {
        queue.add(childNode.parent[i]);
        printNode(childNode.parent[i], childNode); 
      }
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
}