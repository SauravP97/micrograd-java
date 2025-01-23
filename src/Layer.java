import java.util.ArrayList;
import java.util.List;

/** A class representing collection of Neuron in a single Layer. */
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