import java.util.ArrayList;
import java.util.List;

/**
 * Created by ej on 4/21/2017.
 */
public class Layer {

    private List<Neuron> neurons;

    public Layer(int numNeurons, int previousLayerSize){
        neurons = new ArrayList<>(numNeurons);

        for (int i = 0; i < numNeurons; i++){
            neurons.add(new Neuron(previousLayerSize));
        }
    }

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
    public int getHighestValue(){
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
