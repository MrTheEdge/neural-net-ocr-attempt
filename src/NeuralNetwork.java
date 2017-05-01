import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * E.J. Schroeder
 * Matt Moellman
 * AI Program 4
 *
 * Neural Network is the main class of the network. It contains all the the layers and handles the training, testing,
 * forward feeding and back propagation.
 */
public class NeuralNetwork {

    final static int IMG_WIDTH = 8;
    final static int IMG_HEIGHT = 8;
    final static int IMG_SIZE = IMG_WIDTH * IMG_HEIGHT;

    private List<Layer> layers;
    private int inputs;
    private int outputs;
    private int hiddenLayers;
    private int layerSize;
    private double learningRate;

    public NeuralNetwork(int inputs, int outputs, int hiddenLayers, int layerSize, double learningRate){

        this.inputs = inputs;
        this.outputs = outputs;
        this.hiddenLayers = hiddenLayers;
        this.layerSize = layerSize;
        this.learningRate = learningRate;

        layers = new ArrayList<>(hiddenLayers + 2);
        layers.add(new Layer(inputs, 0));
        int prevLayerSize = IMG_SIZE;

        for (int i = 0; i < hiddenLayers; i++){ // Create hidden layers
            layers.add(new Layer(layerSize, prevLayerSize));
            prevLayerSize = layerSize;
        }

        layers.add(new Layer(outputs, layerSize));

    }

    // Propagates forward through the network given a set of initial inputs.
    private void forwardPropagate(List<Double> inputs){
        List<Neuron> inputLayer = layers.get(0).getNeurons(); // Get input layer
        for (int i = 0; i < inputLayer.size(); i++){
            inputLayer.get(i).setValue(inputs.get(i)); // Set the value of each input neuron
        }

        for (int i = 1; i < layers.size(); i++){ // Go thorough each layer
            for (Neuron neuron : layers.get(i).getNeurons()){ // Get the neurons in each layer
                neuron.calculateValue(layers.get(i - 1).getNeurons()); // Calc value for each neuron given previous layer
            }
        }

//        List<Neuron> outputLayer = layers.get(layers.size() - 1).getNeurons();
//        for (Neuron n : outputLayer){
//            System.out.print(n.getValue() + " Summed Value: " + n.outputSum + " ");
//        }
//        System.out.println();
    }

    private void backPropagate(List<Double> expected){
        // Starting with output layer...
        Layer outputLayer = layers.get(layers.size() - 1);
        List<Neuron> neurons = outputLayer.getNeurons();

        for (int i = 0; i < neurons.size(); i++){ // Calculate delta on output neurons
            neurons.get(i).calculateDeltaFromExpected(expected.get(i));
        }

        // Starting from last hidden layer, iterate backwards to first hidden, avoid input layer.
        for (int i = layers.size() - 2; i > 0; i--){
            List<Neuron> layer = layers.get(i).getNeurons();
            for (int j = 0; j < layer.size(); j++){ // For each neuron in the layer
                double error = 0;
                for (Neuron n : layers.get(i + 1).getNeurons()){
                    error += n.calcError(j); // Sum error * weight for each edge connecting this neuron to next layer
                }
                layer.get(j).calculateDelta(error);
            }
        }
    }

    private void updateWeights(double learningRate){

        for (int i = 1; i < layers.size(); i++){ // Ignore input nodes, update weights on all other neurons
            layers.get(i).updateWeights(layers.get(i-1), learningRate);
        }
    }

    public void trainNetwork(int numEpochs) throws IOException {
        for (int epoch = 0; epoch < numEpochs; epoch++){
            double sumError = 0;

            Path path = Paths.get("optdigits.tra"); // Training file
            Stream<String> lineStream = Files.lines(path);
            List<String[]> data = lineStream.map(s -> s.split(",")).collect(Collectors.toList());
            for (int imgNum = 0; imgNum < data.size(); imgNum++){
                String[] lineArray = data.get(imgNum);
                int expectedInt = Integer.parseInt(lineArray[lineArray.length - 1]);
                if (expectedInt > 2) continue; // Only train on 0, 1, or 2

                List<Double> inputs = new ArrayList<>(IMG_SIZE);
                for (int i = 0; i < IMG_SIZE; i++){
                    inputs.add(Double.parseDouble(lineArray[i])); // Apply inputs to input nodes
                }
                forwardPropagate(inputs); // Run inputs through
                List<Neuron> outputLayer = layers.get(layers.size() - 1).getNeurons(); // Get result of forward prop

                List<Double> expectedOutputs = createExpectedList(expectedInt, 3); // Get list of expected outputs

                for (int i = 0; i < outputs; i++){ // Simple error calculation to track progress in printouts
                    sumError += Math.pow(expectedOutputs.get(i) - outputLayer.get(i).getValue(), 2);
                }
                backPropagate(expectedOutputs);
                updateWeights(learningRate);
            }

            System.out.println("Epoch: " + epoch + " Error: " + sumError);
        }
    }

    public void testNetwork() throws IOException {
        Path path = Paths.get("optdigits.tes"); // Testing file
        Stream<String> lineStream = Files.lines(path); // Open file by line and split on ','
        List<String[]> data = lineStream.map(s -> s.split(",")).collect(Collectors.toList());

        int attempts = 0;
        int correct = 0;
        for (int imgNum = 0; imgNum < data.size(); imgNum++){
            String[] lineArray = data.get(imgNum);

            int expected = Integer.parseInt(lineArray[lineArray.length - 1]); // Last value of input is expected result
            if (expected > 2) continue; // Don't test numbers except for 0, 1, 2
            else attempts++;

            List<Double> inputs = new ArrayList<>(IMG_SIZE);
            for (int i = 0; i < IMG_SIZE; i++){
                // Parse string inputs to doubles, then add to list to pass to network
                inputs.add(Double.parseDouble(lineArray[i]));
            }
            forwardPropagate(inputs);

            // Get the highest value of the resulting output neuron values
            int index = layers.get(layers.size() - 1).getHighestIndex();
            if (index == expected)
                correct++;
        }

        System.out.println("After testing, the neural network got " + correct + "/" + attempts + " correct.");
    }

    private List<Double> createExpectedList(int expectedInt, int size){
        List<Double> expected = new ArrayList<>(size);
        for (int i = 0; i < size; i++){
            if (i == expectedInt)
                expected.add(1.0); // Only 1 should be the expected number
            else
                expected.add(0.0);
        }
        return expected;
    }

    public static void main(String[] args){
        // Initialize NN then train it and test.
        NeuralNetwork nn = new NeuralNetwork(IMG_SIZE, 3, 1, (IMG_SIZE + 3) / 2, 0.05);
        try {
            nn.trainNetwork(500);
            nn.testNetwork();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
