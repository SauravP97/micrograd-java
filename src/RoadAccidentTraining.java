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

public class RoadAccidentTraining {
    public static void main(String[] args) throws IOException {
        // Initialize Neural Net
        MultiLayerPreceptron mlp = new MultiLayerPreceptron(5, new int[]{5, 8, 8, 1});
        Value[][] train = buildTrainingData();
        Value[] actual = buildLabelOutput();
        double learningRate = -0.005;
        int iterations = 11;

        Value[] pred = trainMicrograd(train, actual, learningRate, iterations, mlp);
    }

    private static Value[] trainMicrograd(
        Value[][] train, Value[] actual, double learningRate, int iterations, MultiLayerPreceptron mlp) {
        for (int x = 0; x < iterations; x++) {
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
                System.out.println("Gradient: " + parameter.grad + " " +  parameter.label);
                parameter.value += learningRate * parameter.grad;
            }

            if (x == iterations - 1) {
                return pred;
            }
        }

        return null;
    }

    private static Value[][] buildTrainingData() throws IOException {
        Value[][] train = new Value[200][5];
        BufferedReader br = new BufferedReader(new FileReader("../dataset/road_accident_data.csv"));
        int index = 0;
        String line = "";

        while ((line = br.readLine()) != null) {  
            String[] row = line.split(",");
            double age = Double.parseDouble(row[1]);
            double gender = Double.parseDouble(row[2]);
            double speedOfImpact = Double.parseDouble(row[3]);
            double helmetUsed = Double.parseDouble(row[4]);
            double seatbeltUsed = Double.parseDouble(row[5]);
            //System.out.println(age + ", " + gender + ", " + speedOfImpact + ", " + helmetUsed + ", " + seatbeltUsed);

            train[index] = new Value[]{
                new Value(age, "age"), 
                new Value(gender, "gender"),
                new Value(speedOfImpact, "speedOfImpact"),
                new Value(helmetUsed, "helmetUsed"),
                new Value(seatbeltUsed, "seatbeltUsed")
            };
            
            index++;
        }

        return train;
    }

    private static Value[] buildLabelOutput() throws IOException {
        Value[] y = new Value[200];
        BufferedReader br = new BufferedReader(new FileReader("../dataset/road_accident_data.csv"));
        int index = 0;
        String line = "";

        while ((line = br.readLine()) != null) {  
            String[] row = line.split(",");
            double survived = Double.parseDouble(row[6]) == 1.0 ? 1.0 : -1.0;
            // System.out.println(survived);
            y[index] = new Value(survived, "survived");
            index++;
        }

        return y;
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