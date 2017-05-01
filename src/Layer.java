import java.util.ArrayList;
import java.util.List;

/**
 * E.J. Schroeder
 * Matt Moellman
 * AI Program 4
 *
 * A Layer is used to hold Neurons. The Layer class does not hold much logic, with the exception of updating the
 * weights on each of its neurons. It also has a convenience method that returns the index of the highest output value,
 * effectively giving you the answer of which digit the NN recognized.
 */
public class Layer {

    private List<Neuron> neurons;

    // Given the size of this layer, and how many neurons were in the previous, create the necessary number of
    // neurons and weights for each.
    public Layer(int numNeurons, int previousLayerSize){
        neurons = new ArrayList<>(numNeurons);

        for (int i = 0; i < numNeurons; i++){
            neurons.add(new Neuron(previousLayerSize));
        }
    }

    // Given a set of inputs and a learning rate, iterate through and update weights of neurons using their
    // delta values.
    public void updateWeights(Layer inputs, double learningRate){
        List<Neuron> prevLayer = inputs.getNeurons();

        for (Neuron neuron : neurons){ // For each neuron in this layer
            for (int i = 0; i < neuron.numWeights(); i++){ // For each of it's weights
                // Update the weights at index i with the input from previous layer
                neuron.updateWeight(i, prevLayer.get(i).getValue(), learningRate);
            }
            // Update the bias weight
            neuron.updateBias(learningRate);
        }
    }

    // Returns the index of the neuron with the highest value
    // In our case, the index directly represents the number we are trying to recognize
    public int getHighestIndex(){
        int maxIndex = 0;
        double maxValue = Double.MIN_VALUE;

        for (int i = 0; i < neurons.size(); i++){
            if (neurons.get(i).getValue() > maxValue){
                maxIndex = i;
                maxValue = neurons.get(i).getValue();
            }
        }
        return maxIndex;
    }

    public List<Neuron> getNeurons(){
        return neurons;
    }

}
