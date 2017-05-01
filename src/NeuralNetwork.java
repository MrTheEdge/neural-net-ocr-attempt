import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static java.nio.file.Files.lines;

/**
 * Created by ej on 4/21/2017.
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

    public final List<Double> zeroOutput;
    public final List<Double> oneOutput;
    public final List<Double> twoOutput;

    public NeuralNetwork(int inputs, int outputs, int hiddenLayers, int layerSize, double learningRate){

        this.inputs = inputs;
        this.outputs = outputs;
        this.hiddenLayers = hiddenLayers;
        this.layerSize = layerSize;
        this.learningRate = learningRate;

        zeroOutput = new ArrayList<>(3);
        zeroOutput.add(1.0);
        zeroOutput.add(0.0);
        zeroOutput.add(0.0);

        oneOutput = new ArrayList<>(3);
        oneOutput.add(0.0);
        oneOutput.add(1.0);
        oneOutput.add(0.0);

        twoOutput = new ArrayList<>(3);
        twoOutput.add(0.0);
        twoOutput.add(0.0);
        twoOutput.add(1.0);

        layers = new ArrayList<>(hiddenLayers + 2);
        layers.add(new Layer(inputs, 0));
        int prevLayerSize = IMG_SIZE;

        for (int i = 0; i < hiddenLayers; i++){
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

        for (int i = 0; i < neurons.size(); i++){
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

    private void updateWeights(List<Double> inputs, double learningRate){

        for (int i = 1; i < layers.size(); i++){
            layers.get(i).updateWeights(layers.get(i-1), learningRate);
        }
    }

    public void trainNetwork(int numEpochs) throws IOException {
        for (int epoch = 0; epoch < numEpochs; epoch++){
            double sumError = 0;

            Path path = Paths.get("optdigits.tra");
            Stream<String> lineStream = Files.lines(path);
            List<String[]> data = lineStream.map(s -> s.split(",")).collect(Collectors.toList());
            for (int imgNum = 0; imgNum < data.size(); imgNum++){
                String[] lineArray = data.get(imgNum);
                int expectedInt = Integer.parseInt(lineArray[lineArray.length - 1]);
                if (expectedInt > 2) continue;

                List<Double> inputs = new ArrayList<>(IMG_SIZE);
                for (int i = 0; i < IMG_SIZE; i++){
                    inputs.add(Double.parseDouble(lineArray[i]));
                }
                forwardPropagate(inputs);
                List<Neuron> outputLayer = layers.get(layers.size() - 1).getNeurons();

                List<Double> expectedOutputs = createExpectedList(expectedInt);

                for (int i = 0; i < outputs; i++){
                    sumError += Math.pow(expectedOutputs.get(i) - outputLayer.get(i).getValue(), 2);
                }
                backPropagate(expectedOutputs);
                updateWeights(inputs, learningRate);
            }

            System.out.println("Epoch: " + epoch + " Error: " + sumError);
        }
    }

    public void testNetwork() throws IOException {
        Path path = Paths.get("optdigits.tes");
        Stream<String> lineStream = Files.lines(path);
        List<String[]> data = lineStream.map(s -> s.split(",")).collect(Collectors.toList());

        int attempts = 0;
        int correct = 0;
        for (int imgNum = 0; imgNum < data.size(); imgNum++){
            String[] lineArray = data.get(imgNum);

            int expected = Integer.parseInt(lineArray[lineArray.length - 1]);
            if (expected > 2) continue;
            else attempts++;

            List<Double> inputs = new ArrayList<>(IMG_SIZE);
            for (int i = 0; i < IMG_SIZE; i++){
                inputs.add(Double.parseDouble(lineArray[i]));
            }
            forwardPropagate(inputs);

            int index = layers.get(layers.size() - 1).getHighestIndex();
            if (index == expected)
                correct++;
        }

        System.out.println("After testing, the neural network got " + correct + "/" + attempts + " correct.");
    }

    private List<Double> createExpectedList(int expectedInt){
        List<Double> expected = new ArrayList<>(3);
        for (int i = 0; i < 3; i++){
            if (i == expectedInt)
                expected.add(1.0);
            else
                expected.add(0.0);
        }
        return expected;
    }

    public static void main(String[] args){

        NeuralNetwork nn = new NeuralNetwork(IMG_SIZE, 3, 1, (IMG_SIZE + 3) / 2, 0.05);
        try {
            nn.trainNetwork(200);
            nn.testNetwork();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
