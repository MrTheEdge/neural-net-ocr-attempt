import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Created by ej on 4/21/2017.
 */
public class Neuron {

    private List<Double> weights;
    private double biasWeight;

    private double value;           // Store value after going through activation function
    private double outputSum;       // The sum of input values * weights before going through activation
    private double delta;           // The delta used for back propagation from output nodes

    public Neuron(int numInputs){
        if (numInputs > 0){
            // If no inputs, this is an input node
            weights = new ArrayList<>(numInputs);

            Random rand = new Random();

            biasWeight = rand.nextDouble() * 2 - 1;
            for (int i = 0; i < numInputs; i++){
                weights.add(rand.nextDouble() * 2 + 1);
            }
        }

    }

    // Used for input nodes to set the value directly.
    public void setValue(double value){
        this.value = value;
    }

    public double getValue(){
        return value;
    }

    // Sum up all weights * values and pass the result through activation function
    // Assumes that the list of values passed in is the same length as the list of weights
    public void calculateValue(List<Neuron> values){
        double sum = biasWeight; // 1 * biasWeight
        for (int i = 0; i < weights.size() - 1; i++){
            sum += values.get(i).getValue() * weights.get(i);
        }

        outputSum = sum;
        value = sigmoid(outputSum);
    }

    public void updateBias(double learningRate){
        biasWeight += (learningRate * delta);
    }

    public int numWeights(){
        return weights.size();
    }

    public void updateWeight(int index, double input, double learningRate){
        double currentWeight = weights.get(index);
        currentWeight += learningRate * input * delta;
        weights.set(index, currentWeight);
    }

    private double sigmoid(double outputSum) {
        return 1 / (1 + Math.pow(Math.E, -outputSum));
    }

    private double sigmoidPrime(double sum){
        return Math.pow(Math.E, sum) / Math.pow(1 + Math.pow(Math.E, sum), 2);
    }

    public void calculateDelta(double errorSum){
        this.delta = errorSum * sigmoidPrime(value);
    }

    public void calculateDeltaFromExpected(double expected){
        this.delta = (expected - value) * sigmoidPrime(value);
    }

    public double calcError(int weightIndex) {
        return delta * weights.get(weightIndex);
    }
}
