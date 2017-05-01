import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * E.J. Schroeder
 * Matt Moellman
 * AI Program 4
 *
 * Neuron is used to store the data associated with each neuron in the network. Each neuron maintains its own set of
 * weights, along with a bias weight. It keeps its value, but also the summed value of all of its weights before
 * passing it through the activation function. It also keeps an error delta when back propagating so that it can
 * update its weights.
 */
public class Neuron {

    private List<Double> weights;
    private double biasWeight;

    private double value;           // Store value after going through activation function
    public double outputSum;       // The sum of input values * weights before going through activation
    private double delta;           // The delta used for back propagation from output nodes

    public Neuron(int numInputs){
        // If no inputs, this is an input node
        if (numInputs > 0){

            weights = new ArrayList<>(numInputs);

            Random rand = new Random();

            // Set biasWeight and weights to random values between -1 and 1
            biasWeight = rand.nextDouble() * 2 - 1;
            for (int i = 0; i < numInputs; i++){
                weights.add(rand.nextDouble() * 2 - 1);
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

    // Basically the same as updateWeight, except for the bias weight. Used in error propagation
    public void updateBias(double learningRate){
        biasWeight += (learningRate * delta);
    }

    public int numWeights(){
        return weights.size();
    }

    // Given an index for the weight to update, along with the expected value and a learning rate, update the weight
    // to its new value
    public void updateWeight(int index, double input, double learningRate){
        double currentWeight = weights.get(index);
        currentWeight += learningRate * input * delta;
        weights.set(index, currentWeight);
    }

    // The sigmoid function as defined in the slides and elsewhere
    private double sigmoid(double outputSum) {
        return 1 / (1 + Math.pow(Math.E, -outputSum));
    }

    // The derivative of the sigmoid function, used in back propagation
    private double sigmoidPrime(double sum){
        return Math.pow(Math.E, sum) / Math.pow(1 + Math.pow(Math.E, sum), 2);
    }

    // Calculates the error delta of this node based on the combined errors of the layer ahead
    public void calculateDelta(double errorSum){
        this.delta = errorSum * sigmoidPrime(value);
    }

    // Used on output nodes, because calculating the error is as simple as subtracting the value from the expected
    public void calculateDeltaFromExpected(double expected){
        this.delta = (expected - value) * sigmoidPrime(value);
    }

    // Returns delta * weight of a given index for error computations
    public double calcError(int weightIndex) {
        return delta * weights.get(weightIndex);
    }
}
