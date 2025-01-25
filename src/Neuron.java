import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;

/** A Neuron class representing a single Neuron in the Neural Net. */
class Neuron {
  Value[] weights;
  Value bias;

  Neuron(int neuronId, int layerId, int inputs) {
    weights = new Value[inputs];
    for (int i=0; i<inputs; i++) {
      weights[i] = new Value(Math.random(), "w"+Integer.toString(i+1) + "For Layer: " + layerId + "For Neuron: " + neuronId);
    }
    bias = new Value(Math.random(), "b");
  }

  Value activate(Value[] x) {
    Value activatedValue = new Value(0.0, "output");

    for (int i = 0; i < x.length; i++) {
      activatedValue = activatedValue.add(
        weights[i].multiply(
          x[i], "x" + Integer.toString(i+1) + "w" + Integer.toString(i+1)), 
        "z");
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