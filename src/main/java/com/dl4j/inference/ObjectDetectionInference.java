package com.dl4j.inference;

import com.amazonaws.auth.AWSStaticCredentialsProvider;
import com.amazonaws.auth.profile.ProfileCredentialsProvider;
import com.amazonaws.regions.Regions;
import com.amazonaws.services.s3.AmazonS3;
import com.amazonaws.services.s3.AmazonS3ClientBuilder;
import com.dl4j.server.HttpStreamServer;
import com.dl4j.utils.JavaCVHelper;

import java.time.Instant;
import java.util.concurrent.TimeUnit;
import org.bytedeco.javacpp.opencv_core.Point;
import org.bytedeco.javacpp.opencv_core.Scalar;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.javacv.IPCameraFrameGrabber;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Properties;
import java.util.List;
import java.util.Optional;

import static org.bytedeco.javacpp.opencv_core.FONT_HERSHEY_DUPLEX;
import static org.bytedeco.javacpp.opencv_imgproc.putText;
import static org.bytedeco.javacpp.opencv_imgproc.rectangle;

/**
 * https://github.com/klevis/AutonomousDriving/blob/master/src/main/java/ramo/klevis/TinyYoloPrediction.java
 * https://stackoverflow.com/questions/14070370/how-to-capture-and-record-video-from-webcam-using-javacv
 * */
public class ObjectDetectionInference {

    private static Logger log =
            LoggerFactory.getLogger(ObjectDetectionInference.class);

    private static HttpStreamServer httpStreamService;

    private volatile Mat[] v = new Mat[1];
    private ArrayList<DetectedObject> predictedObjects;
    private ComputationGraph model;
    private HashMap<Integer, String> labels;
    private int port;
    private int skipFrame;
    private String s3BucketName;
    private String s3KeyName;
    private AmazonS3 s3Client;
    private String predictionFileName;

    public ObjectDetectionInference(ComputationGraph model, File hyperParametersFile)
            throws IOException {

        final Properties hyperParameters = new Properties();
        hyperParameters.load(new FileInputStream(hyperParametersFile));

        this.predictionFileName = "predictions-" + Instant.now().toEpochMilli() + ".csv";

        this.model = model;
        this.port = Integer.parseInt(hyperParameters.getProperty("port"));
        this.skipFrame = Integer.parseInt(hyperParameters.getProperty("skipFrame"));
        this.s3BucketName = hyperParameters.getProperty("s3BucketName");
        this.s3KeyName = hyperParameters.getProperty("s3KeyName");

        this.s3Client = AmazonS3ClientBuilder.standard()
                .withCredentials(new AWSStaticCredentialsProvider(
                        new ProfileCredentialsProvider().getCredentials()))
                .withRegion(Regions.DEFAULT_REGION)
                .build();

        prepareLabels();
        prepareS3Bucket();
    }

    public void startRealTimeVideoDetection(String videoPath)
            throws Exception {

        FrameGrabber frameGrabber;

        if(videoPath.startsWith("http")) {
            frameGrabber = new IPCameraFrameGrabber(videoPath, 60, 60, TimeUnit.SECONDS);
            frameGrabber.setFormat("mp4");
        } else {
            frameGrabber = new FFmpegFrameGrabber(videoPath);
        }

        frameGrabber.start();

        final double totalFrames = frameGrabber.getLengthInFrames();
        final double frameRate = frameGrabber.getFrameRate();

        log.debug("The input video clip has " + totalFrames + " frames");
        log.debug("The input video clip has frame rate of " + frameRate);

        Frame frame;

        try {

            httpStreamService = new HttpStreamServer(port, v[0]);
            new Thread(httpStreamService).start();

            for(int i = 1; i < totalFrames; i+=skipFrame) {

                frameGrabber.setFrameNumber(i);
                frame = frameGrabber.grab();

                // saveFrameAsDataset(frame, 832, 416, "data/Images-new/frame-" + i + ".jpg"); // for dataset preparation

                JavaCVHelper.saveFrameToJPG(frame, "frame.jpg"); // for prediction usage

                v[0] = new OpenCVFrameConverter.ToMat().convert(frame);

                if(v[0] == null) {
                    log.debug("frame is null");
                    return;
                }

                 markObjectWithBoundingBox(
                         model,
                         v[0],
                         frame.imageWidth,
                         frame.imageHeight,
                         frame.timestamp,
                        true);

                JavaCVHelper.saveMatToJPG(v[0], frame.imageWidth, frame.imageHeight,
                        "frame-prediction.jpg");

                // Push to stream
                httpStreamService.frame= v[0];
            }

            httpStreamService.close();
        } catch(IOException ioe) {
            log.error("Error saving to image file", ioe);
            ioe.printStackTrace();
        }
    }

    private void prepareLabels() {
        final String sponsors = "<USE YOUR OWN LABELS>";
        final String[] sponsorsList = sponsors.split(",");

        int i=0;
        labels = new HashMap<>();
        for(String sponsor: sponsorsList) {
            labels.put(i++, sponsor);
        }
        log.debug(labels.toString());
    }

    private void prepareS3Bucket() {
        if(!s3Client.doesBucketExistV2(s3BucketName)) {
            s3Client.createBucket(s3BucketName);
        }
    }

    private void appendPrediction(String content) {
        PrintWriter out = null;
        try {
            out = new PrintWriter(
                    new BufferedWriter(new FileWriter(
                            this.predictionFileName, true)));
            out.println(content);
            out.flush();
        } catch(IOException ioe) {
            log.error("Error saving prediction", ioe);
        } finally {
            out.close();
        }
    }

    private INDArray prepareImageToArray(
            Mat matFile,
            int width,
            int height) throws IOException {

        final NativeImageLoader nativeImageLoader =
                new NativeImageLoader(height, width, 3);
        final INDArray indArray = nativeImageLoader.asMatrix(matFile);
        JavaCVHelper.saveIndArrayToJPG(indArray, "downsize.jpg");

        return indArray;
    }

    private void markObjectWithBoundingBox(
            ComputationGraph model,
            Mat matFile,
            int imageWidth,
            int imageHeight,
            long imageTimestamp,
            boolean newBoundingBox) throws Exception {

        // this set of dimension is use for 1920x1048 only
        final int scaledToWidth = 832;
        final int scaledToHeight = 416;
        final int gridWidth = 26;
        final int gridHeight = 13;
        final double detectionThreshold = 0.5;

        final Yolo2OutputLayer outputLayer =
                (Yolo2OutputLayer) model.getOutputLayer(0);

        if(newBoundingBox) {
            final INDArray imageOutput = model.outputSingle(
                    prepareImageToArray(matFile, scaledToWidth, scaledToHeight));

            predictedObjects = (ArrayList<DetectedObject>)
                    outputLayer.getPredictedObjects(imageOutput, detectionThreshold);
            log.debug("Total predicted objects: " + predictedObjects.toString());
        }

        markWithBoundingBox(
                matFile,
                gridWidth,
                gridHeight,
                imageWidth,
                imageHeight,
                imageTimestamp);
    }

    private void markWithBoundingBox(
            Mat matFile,
            int gridWidth,
            int gridHeight,
            int imageWidth,
            int imageHeight,
            long imageTimestamp) {

        if (predictedObjects != null) {

            final ArrayList<DetectedObject> detectedObjects =
                    new ArrayList<>(predictedObjects);

            while(!detectedObjects.isEmpty()) {

                Optional<DetectedObject> maxDetectedObjectOptional =
                        detectedObjects.stream().max((detectedObject1, detectedObject2) ->
                                ((Double) detectedObject1.getConfidence())
                                        .compareTo(detectedObject2.getConfidence())
                        );

                if(maxDetectedObjectOptional.isPresent()) {

                    DetectedObject maxDetectedObject = maxDetectedObjectOptional.get();
                    removeObjectsIntersectingWithMax(detectedObjects, maxDetectedObject);
                    detectedObjects.remove(maxDetectedObject);

                    markWithBoundingBox(
                            matFile,
                            gridWidth,
                            gridHeight,
                            imageWidth,
                            imageHeight,
                            imageTimestamp,
                            maxDetectedObject);
                }
            }
        }
    }

    private void markWithBoundingBox(
            Mat matFile,
            int gridWidth,
            int gridHeight,
            int imageWidth,
            int imageHeight,
            long imageTimestamp,
            DetectedObject detectedObject) {

        double[] xy1 = detectedObject.getTopLeftXY();
        double[] xy2 = detectedObject.getBottomRightXY();
        int predictedClass = detectedObject.getPredictedClass();
        double confidence = detectedObject.getConfidence();

        int x1 = (int) Math.round(imageWidth * xy1[0] / gridWidth);
        int y1 = (int) Math.round(imageHeight * xy1[1] / gridHeight);
        int x2 = (int) Math.round(imageWidth * xy2[0] / gridWidth);
        int y2 = (int) Math.round(imageHeight * xy2[1] / gridHeight);

        rectangle(matFile,
                new Point(x1, y1),
                new Point(x2, y2),
                Scalar.GREEN);

        putText(matFile, labels.get(predictedClass) + ", " + Math.round(confidence*100) + "%",
                new Point(x1 + 2, y2 - 2),
                FONT_HERSHEY_DUPLEX, 1,
                Scalar.GREEN);

        appendPrediction(
                labels.get(predictedClass) +
                        ";" + Math.round(confidence*100) +
                        ";" + imageTimestamp);

        s3Client.putObject(s3BucketName, s3KeyName, new File(this.predictionFileName));
    }

    private void removeObjectsIntersectingWithMax(
            ArrayList<DetectedObject> detectedObjects,
            DetectedObject maxObjectDetect) {

        double[] bottomRightXYMax = maxObjectDetect.getBottomRightXY();
        double[] topLeftXYMax = maxObjectDetect.getTopLeftXY();

        List<DetectedObject> removeIntersectingObjects = new ArrayList<>();

        for (DetectedObject detectedObject : detectedObjects) {

            double[] topLeftXY = detectedObject.getTopLeftXY();
            double[] bottomRightXY = detectedObject.getBottomRightXY();

            double iox1 = Math.max(topLeftXY[0], topLeftXYMax[0]);
            double ioy1 = Math.max(topLeftXY[1], topLeftXYMax[1]);

            double iox2 = Math.min(bottomRightXY[0], bottomRightXYMax[0]);
            double ioy2 = Math.min(bottomRightXY[1], bottomRightXYMax[1]);

            double inter_area = (ioy2 - ioy1) * (iox2 - iox1);

            double box1_area =
                    (bottomRightXYMax[1] - topLeftXYMax[1]) * (bottomRightXYMax[0] - topLeftXYMax[0]);
            double box2_area =
                    (bottomRightXY[1] - topLeftXY[1]) * (bottomRightXY[0] - topLeftXY[0]);

            double union_area = box1_area + box2_area - inter_area;
            double iou = inter_area / union_area;

            if (iou > 0.5) {
                removeIntersectingObjects.add(detectedObject);
            }
        }

        detectedObjects.removeAll(removeIntersectingObjects);
    }

    public static void main(String[] args) throws Exception {
        final File hyperParameters = new File(args[2]);
        new ObjectDetectionInference(
                ModelSerializer.restoreComputationGraph(args[0]), hyperParameters)
                .startRealTimeVideoDetection(args[1]);
    }
}