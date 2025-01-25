import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.Queue;
import java.io.BufferedReader;  
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;  

public class MicroGrad {
  public static void main(String[] args) throws IOException {
    // Initialize Neural Net
    MultiLayerPreceptron mlp = new MultiLayerPreceptron(2, new int[]{16, 16, 1});

    Value[][] train = buildTrainingData();
    Value[] actual = buildLabelOutput();
    double learningRate = -0.01;

    // Iterations
    for (int x = 0; x<1000; x++) {
      Value[] pred = new Value[actual.length];
      Value netLoss = new Value(0.0, "netLoss");

      for (int i=0; i<train.length; i++) {
        pred[i] = mlp.activate(train[i])[0];
        Value curLoss1 = pred[i].subtract(actual[i], "loss1");
        Value curLoss2 = pred[i].subtract(actual[i], "loss2");
        Value curLoss = curLoss1.multiply(curLoss2, "loss");

        netLoss = netLoss.add(curLoss, "netLoss");
      }
    
      System.out.println("Net Loss at iteration: " + x);
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
        // System.out.println("Gradient: " + parameter.grad + " " +  parameter.label);
        parameter.value += learningRate * parameter.grad;
      }

      if (x == 999) {
        saveFinalPreditcion(pred);
      }
    }
  }

  private static void saveFinalPreditcion(Value[] pred) throws IOException {
    FileWriter myWriter = new FileWriter("../dataset/predictions.txt");
    for (int i=0; i<pred.length; i++) {
      myWriter.write(pred[i].value + "\n");
    }
    myWriter.close();
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

  private static Value[][] buildTrainingData() throws IOException {
    Value[][] train = new Value[100][2];
    BufferedReader br = new BufferedReader(new FileReader("../dataset/train.csv"));
    int index = 0;
    String line = "";

    while ((line = br.readLine()) != null) {  
      String[] row = line.split(",");
      double x1 = Double.parseDouble(row[1]);
      double x2 = Double.parseDouble(row[2]);
      train[index] = new Value[]{new Value(x1, "x1"), new Value(x2, "x2")};
      // System.out.println(x1 + " " + x2);
      index++; 
    }

    return train;
  }

  private static Value[] buildLabelOutput() throws IOException {
    Value[] y = new Value[100];
    BufferedReader br = new BufferedReader(new FileReader("../dataset/train.csv"));
    int index = 0;
    String line = "";

    while ((line = br.readLine()) != null) {  
      String[] row = line.split(",");
      double yLabel = Double.parseDouble(row[3]);
      y[index] = new Value(yLabel, "y");
      // System.out.println(yLabel);
      index++;
    }

    return y;
  }
}