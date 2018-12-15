package com.dl4j.utils;

import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.Java2DFrameConverter;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.datavec.image.loader.Java2DNativeImageLoader;
import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class JavaCVHelper {

    private static Logger log =
            LoggerFactory.getLogger(JavaCVHelper.class);
    
    public static void saveIndArrayToJPG(
            INDArray indArray,
            String fileName) {
        try {
            Java2DNativeImageLoader java2DNativeImageLoader =
                    new Java2DNativeImageLoader();
            ImageIO.write(
                    java2DNativeImageLoader.asBufferedImage(indArray),
                    "jpg", new File(fileName));
        } catch(IOException ioe) {
            log.error("Error saving frame to jpg file", ioe);
            ioe.printStackTrace();
        }
    }

    public static void saveFrameToJPG(
            Frame frame,
            String fileName) {
        try {
            BufferedImage bufferedImage = new Java2DFrameConverter().convert(frame);
            if(bufferedImage != null) {
                ImageIO.write(bufferedImage, "jpg", new File(fileName));
            }
        } catch(IOException ioe) {
            log.error("Error saving frame to jpg file", ioe);
            ioe.printStackTrace();
        }
    }

    public static void saveFrameAsDataset(
            Frame frame,
            int imageWidth,
            int imageHeight,
            String fileName) {
        saveMatToJPG(new OpenCVFrameConverter.ToMat().convert(frame),
                imageWidth, imageHeight, fileName);
    }

    public static BufferedImage saveMatToBufferImage(Mat matFile)
        throws IOException {
        final INDArray indArray =
                new NativeImageLoader().asMatrix(matFile);

        final Java2DNativeImageLoader java2DNativeImageLoader =
                new Java2DNativeImageLoader();

        return java2DNativeImageLoader.asBufferedImage(indArray);
    }

    public static BufferedImage saveMatToBufferImage(
            Mat matFile,
            int imageWidth,
            int imageHeight)
            throws IOException {

        final INDArray indArray =
                new NativeImageLoader(imageHeight, imageWidth, 3)
                        .asMatrix(matFile);

        final Java2DNativeImageLoader java2DNativeImageLoader =
                new Java2DNativeImageLoader();

        return java2DNativeImageLoader.asBufferedImage(indArray);
    }

    public static void saveMatToJPG(
            Mat matFile,
            int imageWidth,
            int imageHeight,
            String fileName) {
        try {
            ImageIO.write(
                    saveMatToBufferImage(matFile, imageWidth, imageHeight),
                    "jpg", new File(fileName));
        } catch(IOException ioe) {
            log.error("Error saving mat file to jpg", ioe);
            ioe.printStackTrace();
        }
    }
}
