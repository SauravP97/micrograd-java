import java.util.ArrayList;
import java.util.List;

/** A Multi Layer Preceptron class representing the collection of Layers. */
public class MultiLayerPreceptron {
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