package com.dl4j.models;

import com.dl4j.utils.Helper;
import com.dl4j.utils.ImageDataHelper;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.InputSplit;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.ImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.saver.LocalFileGraphSaver;
import org.deeplearning4j.earlystopping.scorecalc.ClassificationScoreCalculator;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingGraphTrainer;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.ResNet50;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.Properties;
import java.util.Random;

public class ImageClassificationModel {

    private static Logger log =
            LoggerFactory.getLogger(ImageClassificationModel.class);

    private int totalLabels;
    private Random rand;
    private File modelFile;
    private File dataFile;
    private File hyperParametersFile;
    private ComputationGraph graph;
    private ParentPathLabelGenerator labelMaker;
    private InputSplit trainData;
    private InputSplit testData;

    public ImageClassificationModel(final File dataFile, final File hyperParametersFile) {
        this.rand = new Random(ImageDataHelper.generateSeed(100, 999));
        this.dataFile = dataFile;
        this.hyperParametersFile = hyperParametersFile;
        this.labelMaker = new ParentPathLabelGenerator();
    }

    public ImageClassificationModel(
            final File dataFile,
            final File hyperParametersFile,
            final File modelFile) {
        this.rand = new Random(ImageDataHelper.generateSeed(100, 999));
        this.modelFile = modelFile;
        this.dataFile = dataFile;
        this.hyperParametersFile = hyperParametersFile;
        this.labelMaker = new ParentPathLabelGenerator();
    }

    /**
     * Pre-process the data into train and test data.
     * Precondition: download dataset to resource folder first.
     * See https://www.kaggle.com/jessicali9530/caltech256 for data folder structure.
     */
    public void data() {
        final int maxImagesPerLabel = 80;
        final double trainPartitionPercentage = 0.8;

        log.debug("Randomize image files, split to train and test datasets...");
        final InputSplit[] inputSplits =
                ImageDataHelper.splitImageDataToTrainAndTest(
                        rand, dataFile, labelMaker, maxImagesPerLabel,
                        trainPartitionPercentage);

        trainData = inputSplits[0];
        testData = inputSplits[1];
        totalLabels = ImageDataHelper.getTotalLabels(dataFile);
    }

    public void fit() throws IOException {

        final Properties hyperParameters = new Properties();
        hyperParameters.load(new FileInputStream(hyperParametersFile));

        final int imageHeight = 224;
        final int imageWidth = 224;
        final int rgbChannels = 3;
        final int batchSize = Integer.parseInt(hyperParameters.getProperty("batchSize"));
        final int epochs = Integer.parseInt(hyperParameters.getProperty("epochs"));
        final int totalOutputClasses = Integer.parseInt(hyperParameters.getProperty("totalOutputClasses"));
        final double learningRate = Double.parseDouble(hyperParameters.getProperty("learningRate"));
        final String exportModelFileName = hyperParameters.getProperty("exportModelFileName");

        log.debug("batchSize: " + batchSize);
        log.debug("epochs: " + epochs);
        log.debug("totalOutputClasses: " + totalOutputClasses);
        log.debug("learningRate: " + learningRate);

        final boolean imageTransformShuffle = true;
        final double imageTransformProbability1 = 0.9;
        final double imageTransformProbability2 = 0.7;

        final ImageRecordReader recordReader =
                new ImageRecordReader(imageHeight, imageWidth, rgbChannels, labelMaker);

        DataSetIterator dataIter;

        if(modelFile != null) {
            log.debug("Restore existing computation graph...");
            graph = ModelSerializer.restoreComputationGraph(modelFile);
        } else {
            log.debug("Create new computation graph from zoo model...");
            final ZooModel zooModel = ResNet50.builder()
                    .numClasses(totalOutputClasses)
                    .build();

            final ComputationGraph computationGraph = (ComputationGraph) zooModel.initPretrained();
            log.debug(computationGraph.summary());

            final FineTuneConfiguration fineTuneConfiguration = new FineTuneConfiguration.Builder()
                    .updater(new Adam(learningRate))
                    .seed(Helper.generateSeed(100, 999))
                    .build();

            graph = new TransferLearning.GraphBuilder(computationGraph)
                    .fineTuneConfiguration(fineTuneConfiguration)
                    .setFeatureExtractor("flatten_1")
                    .removeVertexKeepConnections("fc1000")
                    .addLayer("output",
                            new OutputLayer.Builder(
                                    LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                    .nIn(2048)
                                    .nOut(totalOutputClasses)
                                    .activation(Activation.SOFTMAX).build(),
                            "flatten_1")
                    .setOutputs("output")
                    .build();
        }

        log.debug("Transferred learning completed, add an output layer...");
        log.debug(graph.summary());

        Helper.startUIServer(graph);

        log.debug("Train model with no transform and early stopping....");
        // Train without transformations
        recordReader.initialize(trainData, null);
        dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, totalLabels);

        final EarlyStoppingConfiguration earlyStoppingConfiguration = new EarlyStoppingConfiguration.Builder()
                .epochTerminationConditions(
                        new MaxEpochsTerminationCondition(epochs))
                .scoreCalculator(
                        new ClassificationScoreCalculator(Evaluation.Metric.F1, dataIter))
                .evaluateEveryNEpochs(1)
                .modelSaver(
                        new LocalFileGraphSaver(System.getProperty("user.dir")))
                .build();

        EarlyStoppingGraphTrainer earlyStoppingGraphTrainer = new EarlyStoppingGraphTrainer(
                earlyStoppingConfiguration, graph, dataIter);

        EarlyStoppingResult<ComputationGraph> earlyStoppingResult = earlyStoppingGraphTrainer.fit();
        graph = earlyStoppingResult.getBestModel();

        log.debug("Model training with no transform completed...");
        log.debug("Termination reason: " + earlyStoppingResult.getTerminationReason());
        log.debug("Termination details: " + earlyStoppingResult.getTerminationDetails());
        log.debug("Total epoch: " + earlyStoppingResult.getTotalEpochs());
        log.debug("Best model epoch: " + earlyStoppingResult.getBestModelEpoch());
        log.debug("Best model score: " + earlyStoppingResult.getBestModelScore());

        log.debug("Train model with transform and early stopping....");
        final ImageTransform transform = ImageDataHelper.generateImageTransformPipeline(
                rand, new Random(ImageDataHelper.generateSeed(100, 999)),
                imageTransformProbability1,
                imageTransformProbability2,
                imageTransformShuffle);

        recordReader.initialize(trainData, transform);
        dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, totalLabels);

        earlyStoppingGraphTrainer = new EarlyStoppingGraphTrainer(
                earlyStoppingConfiguration, graph, dataIter);

        earlyStoppingResult = earlyStoppingGraphTrainer.fit();
        graph = earlyStoppingResult.getBestModel();

        log.debug("Model training with transform completed...");
        log.debug("Termination reason: " + earlyStoppingResult.getTerminationReason());
        log.debug("Termination details: " + earlyStoppingResult.getTerminationDetails());
        log.debug("Total epoch: " + earlyStoppingResult.getTotalEpochs());
        log.debug("Best model epoch: " + earlyStoppingResult.getBestModelEpoch());
        log.debug("Best model score: " + earlyStoppingResult.getBestModelScore());

        // Updater: i.e., the state for Momentum, RMSProp, Adagrad etc.
        // Save this if you want to train your network more in the future
        final boolean saveUpdater = true;
        ModelSerializer.writeModel(graph, new File(exportModelFileName), saveUpdater);


        log.debug("Train completed...");
        log.debug("Evaluate model....");
        recordReader.initialize(testData);
        dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, totalLabels);

        final Evaluation eval = graph.evaluate(dataIter);
        log.debug(eval.stats());
        log.debug("We are done!!!!!");
    }

    public static void main(String[] args) {
        try {

            final File dataFile = new File(args[0]);
            final File hyperParameters = new File(args[1]);

            ImageClassificationModel model;

            if(args.length == 3) {
                final File modelFile = new File(args[2]);
                model = new ImageClassificationModel(dataFile, hyperParameters, modelFile);
            } else {
                model = new ImageClassificationModel(dataFile, hyperParameters);
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
