package com.dl4j.models;

import com.dl4j.utils.Helper;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.Properties;
import java.util.Random;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator.AlignmentMode;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class TimeSeriesPredictionModel {

    private static Logger log =
            LoggerFactory.getLogger(TimeSeriesPredictionModel.class);

    private Random rand;
    private File modelFile;
    private File dataFile;
    private File hyperParametersFile;
    private SequenceRecordReaderDataSetIterator dataSetIterator;

    public TimeSeriesPredictionModel(
            File dataFile,
            File hyperParametersFile) {
        this.rand = new Random(Helper.generateSeed(100, 999));
        this.dataFile = dataFile;
        this.hyperParametersFile = hyperParametersFile;
    }

    public TimeSeriesPredictionModel(
            File dataFile,
            File hyperParametersFile,
            File modelFile) {
        this.rand = new Random(Helper.generateSeed(100, 999));
        this.dataFile = dataFile;
        this.hyperParametersFile = hyperParametersFile;
        this.modelFile = modelFile;
    }

    public void data() throws InterruptedException, IOException {
        log.debug("Load data...");

        final int numLinesToSkip = 1;
        final String delimiter = ",";

        final CSVSequenceRecordReader featureRecordReader = new CSVSequenceRecordReader(
                numLinesToSkip, delimiter);
        featureRecordReader.initialize(new FileSplit(dataFile));

        final CSVSequenceRecordReader labelRecordReader = new CSVSequenceRecordReader(
                numLinesToSkip+1, delimiter);
        labelRecordReader.initialize(new FileSplit(dataFile));

//        System.out.println(featureRecordReader.sequenceRecord());
//        System.out.println(labelRecordReader.sequenceRecord());

        final int miniBatchSize = 12; // move to hyper parameter properties file
        final int numPossibleLabels = -1;
        final boolean regression = true;

        dataSetIterator = new SequenceRecordReaderDataSetIterator(
                featureRecordReader, labelRecordReader, miniBatchSize, numPossibleLabels,
                regression, AlignmentMode.ALIGN_END);

        //Normalize the training data
        final DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fitLabel(true);
        normalizer.fit(dataSetIterator);
        dataSetIterator.reset();
        dataSetIterator.setPreProcessor(normalizer);
    }

    public void fit() throws IOException {

        final Properties hyperParameters = new Properties();
        hyperParameters.load(new FileInputStream(hyperParametersFile));

        final int hiddenLayers = 200;
        final int timeSeriesLength = 12;

        final int epochs = Integer.parseInt(hyperParameters.getProperty("epochs"));
        final double learningRate = Double.parseDouble(hyperParameters.getProperty("learningRate"));
        final String exportModelFileName = hyperParameters.getProperty("exportModelFileName");

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(Helper.generateSeed(100, 999))
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam(learningRate))
                .weightInit(WeightInit.XAVIER)
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                .gradientNormalizationThreshold(0.5)
                .list()
                .layer(0, new LSTM.Builder()
                        .activation(Activation.TANH)
                        .nIn(1)
                        .nOut(hiddenLayers)
                        .build())
                .layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation(Activation.IDENTITY)
                        .nIn(hiddenLayers)
                        .nOut(1)
                        .build())
                .backpropType(BackpropType.TruncatedBPTT)
                .tBPTTForwardLength(timeSeriesLength)
                .tBPTTBackwardLength(timeSeriesLength)
                .pretrain(false)
                .backprop(true)
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);

        net.init();
        net.setListeners(new ScoreIterationListener(100));

        log.debug(net.summary());

        for (int i = 0; i < epochs; i++) {
            net.fit(dataSetIterator);
            log.debug("*** Completed epoch {} ***", i);

            dataSetIterator.reset();
            net.rnnClearPreviousState();
        }

        // Export
        final File locationToSave = new File(exportModelFileName);

        // Updater: i.e., the state for Momentum, RMSProp, Adagrad etc.
        // Save this if you want to train your network more in the future
        final boolean saveUpdater = true;
        ModelSerializer.writeModel(net, locationToSave, saveUpdater);

        net = ModelSerializer.restoreMultiLayerNetwork(locationToSave);

        INDArray testInput = dataSetIterator.next().getFeatures();
        INDArray output = net.rnnTimeStep(testInput);
        dataSetIterator.reset();

        log.debug(testInput.toString());
        log.debug(output.toString());
        log.debug("----------------------");
    }

    public static void main(String[] args) {
        try {
            final File dataFile = new File(args[0]);
            final File hyperParameters = new File(args[1]);

            TimeSeriesPredictionModel model;

            if (args.length == 3) {
                final File modelFile = new File(args[2]);
                model = new TimeSeriesPredictionModel(dataFile, hyperParameters, modelFile);
            } else {
                model = new TimeSeriesPredictionModel(dataFile, hyperParameters);
            }

            model.data();
            model.fit();
            System.exit(0);
        } catch (Exception e) {
            e.printStackTrace();
            log.error("Error running main()", e);
        }
    }
}
