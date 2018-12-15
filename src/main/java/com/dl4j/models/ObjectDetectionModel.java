package com.dl4j.models;

import com.dl4j.utils.Helper;
import com.dl4j.utils.ImageDataHelper;
import org.datavec.api.io.filters.RandomPathFilter;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.objdetect.ObjectDetectionRecordReader;
import org.datavec.image.recordreader.objdetect.impl.VocLabelProvider;
import org.datavec.image.transform.ImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.objdetect.Yolo2OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.model.YOLO2;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.Properties;
import java.util.Random;

public class ObjectDetectionModel {

    private static Logger log =
            LoggerFactory.getLogger(ObjectDetectionModel.class);

    private Random rand;
    private File modelFile;
    private File dataFile;
    private File hyperParametersFile;
    private ComputationGraph graph;
    private InputSplit trainData;

    public ObjectDetectionModel(File dataFile, File hyperParametersFile) {
        this.rand = new Random(Helper.generateSeed(100, 999));
        this.dataFile = dataFile;
        this.hyperParametersFile = hyperParametersFile;
    }

    public ObjectDetectionModel(
            File dataFile,
            File hyperParametersFile,
            File modelFile) {
        this.rand = new Random(Helper.generateSeed(100, 999));
        this.dataFile = dataFile;
        this.hyperParametersFile = hyperParametersFile;
        this.modelFile = modelFile;
    }

    public void data() {

        log.debug("Load data...");
        final File imageDir = new File(dataFile.getPath(), "Images");
        final RandomPathFilter pathFilter = new RandomPathFilter(rand) {
            @Override
            protected boolean accept(String name) {
                name = name.replace("/Images/", "/Annotations/")
                        .replace(".jpg", ".xml");
                try {
                    return new File(new URI(name)).exists();
                } catch (URISyntaxException ex) {
                    throw new RuntimeException(ex);
                }
            }
        };

        final InputSplit[] data = new FileSplit(imageDir, NativeImageLoader.ALLOWED_FORMATS, rand)
                .sample(pathFilter);
        trainData = data[0];
    }

    public void fit() throws IOException {

        final Properties hyperParameters = new Properties();
        hyperParameters.load(new FileInputStream(hyperParametersFile));

        final int numberOfBoundingBoxes = 5;
        INDArray priorBoxes = Nd4j.create(new double[][] {{1.5, 1.5}, {2, 2}, {3,3}, {3.5, 8}, {4, 9}});

        final double lambdaNoObj = 0.5;
        final double lambdaCoord = 1.0;

        // TODO: Move to properties file
        final int imageWidth = 832;
        final int imageHeight = 416;
        final int rgbChannels = 3;
        final int gridWidth = 26;
        final int gridHeight = 13;

        final int batchSize = Integer.parseInt(hyperParameters.getProperty("batchSize"));
        final int epochs = Integer.parseInt(hyperParameters.getProperty("epochs"));
        final double learningRate = Double.parseDouble(hyperParameters.getProperty("learningRate"));
        final String exportModelFileName = hyperParameters.getProperty("exportModelFileName");

        log.debug("batchSize: " + batchSize);
        log.debug("epochs: " + epochs);
        log.debug("learningRate: " + learningRate);

        final ObjectDetectionRecordReader recordReaderTrain =
                new ObjectDetectionRecordReader(imageHeight, imageWidth, rgbChannels,
                        gridHeight, gridWidth, new VocLabelProvider(dataFile.getPath()));

        recordReaderTrain.initialize(trainData);

        final int totalOutputClasses = recordReaderTrain.getLabels().size();

        log.debug("total train labels: " + recordReaderTrain.getLabels());
        log.debug("total train labels size: " + totalOutputClasses);

        RecordReaderDataSetIterator trainDataSetIterator =
                new RecordReaderDataSetIterator(
                        recordReaderTrain, batchSize, 1, 1, true);

        trainDataSetIterator.setPreProcessor(new ImagePreProcessingScaler(0, 1));

        if (modelFile != null) {
            log.debug("Restore existing computation graph...");
            graph = ModelSerializer.restoreComputationGraph(modelFile);
        } else {
            log.debug("Create new computation graph from zoo model...");
            ComputationGraph pretrained = (ComputationGraph) YOLO2.builder().build().initPretrained();
            log.debug(pretrained.summary());

            final FineTuneConfiguration fineTuneConfiguration = new FineTuneConfiguration.Builder()
                    .seed(Helper.generateSeed(100, 999))
                    .updater(new Adam(1e-3))
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                    .gradientNormalizationThreshold(1.0)
                    .activation(Activation.IDENTITY)
                    .miniBatch(true)
                    .trainingWorkspaceMode(WorkspaceMode.ENABLED)
                    .build();

            graph = new TransferLearning.GraphBuilder(pretrained)
                    .fineTuneConfiguration(fineTuneConfiguration)
                    .setInputTypes(InputType.convolutional(imageHeight, imageWidth, rgbChannels))
                    .removeVertexKeepConnections("conv2d_23")
                    .addLayer("conv2d_23",
                            new ConvolutionLayer.Builder(1, 1)
                                    .nIn(1024)
                                    .nOut(numberOfBoundingBoxes * (5 + totalOutputClasses))
                                    .stride(1, 1)
                                    .convolutionMode(ConvolutionMode.Same)
                                    .weightInit(WeightInit.UNIFORM)
                                    .hasBias(false)
                                    .activation(Activation.IDENTITY)
                                    .cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)
                                    .build(),
                            "leaky_re_lu_22")
                    .addLayer("outputs",
                            new Yolo2OutputLayer.Builder()
                                    .lambbaNoObj(lambdaNoObj)
                                    .lambdaCoord(lambdaCoord)
                                    .boundingBoxPriors(priorBoxes)
                                    .build(),
                            "conv2d_23")
                    .setOutputs("outputs")
                    .build();
        }

        log.debug(graph.summary());

        Helper.startUIServer(graph);

        log.debug("Training with no transform...");
        for (int i = 0; i < epochs; i++) {
            graph.fit(trainDataSetIterator);
            log.debug("*** Completed epoch {} ***", i);
            trainDataSetIterator.reset();
        }

        final boolean imageTransformShuffle = true;
        final double imageTransformProbability1 = 0.9;
        final double imageTransformProbability2 = 0.7;

        final ImageTransform transform = ImageDataHelper.generateImageTransformPipeline(
                rand, new Random(ImageDataHelper.generateSeed(100, 999)),
                imageTransformProbability1,
                imageTransformProbability2,
                imageTransformShuffle);

        recordReaderTrain.initialize(trainData, transform);
        trainDataSetIterator = new RecordReaderDataSetIterator(
                recordReaderTrain, batchSize, 1, 1, true);

        log.debug("Training with transform...");
        for (int i = 0; i < epochs; i++) {
            graph.fit(trainDataSetIterator);
            log.debug("*** Completed epoch {} ***", i);
            trainDataSetIterator.reset();
        }

        // Export
        final File locationToSave = new File(exportModelFileName);

        // Updater: i.e., the state for Momentum, RMSProp, Adagrad etc.
        // Save this if you want to train your network more in the future
        final boolean saveUpdater = true;
        ModelSerializer.writeModel(graph, locationToSave, saveUpdater);

        log.debug("*** Completed train ***");
    }

    public static void main(String[] args) {
        try {
            final File dataFile = new File(args[0]);
            final File hyperParameters = new File(args[1]);

            ObjectDetectionModel model;

            if (args.length == 3) {
                final File modelFile = new File(args[2]);
                model = new ObjectDetectionModel(dataFile, hyperParameters, modelFile);
            } else {
                model = new ObjectDetectionModel(dataFile, hyperParameters);
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
