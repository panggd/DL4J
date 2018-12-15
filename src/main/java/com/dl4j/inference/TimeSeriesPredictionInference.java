package com.dl4j.inference;

import java.io.IOException;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * This is the template to dockerize the inference codes for SageMaker.
 * Use with DL4J-SageMaker.
 * */
public class TimeSeriesPredictionInference {

    private static String mode;
    private static String model;
    private static String data;
    private static MultiLayerNetwork net;

    public static void main(String[] args) throws Exception {
        mode = args[0];
        model = args[1];
        switch(mode) {
            case "invoke":
                data = args[2];
                invoke();
                break;
            case "ping":
            default:
                ping();
                break;
        }
    }

    private static void ping() throws IOException {
        net = ModelSerializer.restoreMultiLayerNetwork(model);
        System.out.println("OK");
    }

    private static void invoke() throws IOException {
        net = ModelSerializer.restoreMultiLayerNetwork(model);
        float[] dataArray = new float[] {Float.parseFloat(data)};
        INDArray input = Nd4j.create(dataArray);
        INDArray output = net.rnnTimeStep(input);
        System.out.println(output);
    }
}
