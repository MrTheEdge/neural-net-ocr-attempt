import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by ej on 4/21/2017.
 */
public class NeuralNetwork {

    final static int IMG_WIDTH = 28;
    final static int IMG_HEIGHT = 28;
    final static int IMG_SIZE = IMG_WIDTH * IMG_HEIGHT;
    final static int NUM_IMAGES = 10;

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
            byte[] data;

            // Test 0
            data = BinaryFileReader.readBinaryFile("data0");
            for (int imgNum = 0; imgNum < 100; imgNum++){
                List<Double> inputs = new ArrayList<>(IMG_SIZE);
                for (int i = 0; i < IMG_SIZE; i++){
                    inputs.add((data[i + imgNum * IMG_SIZE] & 0xff) / 255.0);
                }
                forwardPropagate(inputs);
                List<Neuron> outputLayer = layers.get(layers.size() - 1).getNeurons();
                for (int i = 0; i < outputs; i++){
                    sumError += Math.pow(zeroOutput.get(i) - outputLayer.get(i).getValue(), 2);
                }
                backPropagate(zeroOutput);
                updateWeights(inputs, learningRate);
            }

            // Test 1
            data = BinaryFileReader.readBinaryFile("data1");
            for (int imgNum = 0; imgNum < 100; imgNum++){
                List<Double> inputs = new ArrayList<>(IMG_SIZE);
                for (int i = 0; i < IMG_SIZE; i++){
                    inputs.add((data[i + imgNum * IMG_SIZE] & 0xff) / 255.0);
                }
                forwardPropagate(inputs);
                List<Neuron> outputLayer = layers.get(layers.size() - 1).getNeurons();
                for (int i = 0; i < outputs; i++){
                    sumError += Math.pow(oneOutput.get(i) - outputLayer.get(i).getValue(), 2);
                }
                backPropagate(oneOutput);
                updateWeights(inputs, learningRate);
            }

            // Test 2
            data = BinaryFileReader.readBinaryFile("data2");
            for (int imgNum = 0; imgNum < 100; imgNum++){
                List<Double> inputs = new ArrayList<>(IMG_SIZE);
                for (int i = 0; i < IMG_SIZE; i++){
                    inputs.add((data[i + imgNum * IMG_SIZE] & 0xff) / 255.0);
                }
                forwardPropagate(inputs);
                List<Neuron> outputLayer = layers.get(layers.size() - 1).getNeurons();
                for (int i = 0; i < outputs; i++){
                    sumError += Math.pow(twoOutput.get(i) - outputLayer.get(i).getValue(), 2);
                }
                backPropagate(twoOutput);
                updateWeights(inputs, learningRate);
            }

            System.out.println("Epoch: " + epoch + " Error: " + sumError);
        }
    }

    public static void main(String[] args){
        NeuralNetwork nn = new NeuralNetwork(IMG_SIZE, 3, 1, 350, 0.005);
        try {
            nn.trainNetwork(50);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
