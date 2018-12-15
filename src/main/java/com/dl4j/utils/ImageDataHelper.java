package com.dl4j.utils;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.transform.FlipImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.PipelineImageTransform;

import java.io.File;
import java.util.Random;

public class ImageDataHelper extends Helper {

    public static int getTotalLabels(final File imageDataPath) {
        final FileSplit fileSplit = new FileSplit(
                imageDataPath,
                NativeImageLoader.ALLOWED_FORMATS);

        final int totalLabels =
                fileSplit.getRootDir().listFiles(File::isDirectory).length;

        return totalLabels;
    }

    /**
     * Given a directory that contains subdirectories.
     * Each sub-directory is a label containing image files.
     * Consolidate all image files, randomize and
     * remove random image to balance total image files per label.
     * Split the remaining image files to test and train dataset.
     *
     * @param rand  Random object to random image files.
     * @param imageDataPath The image data path containing the label sub directories and image files.
     * @param labelMaker    The label generator
     * @param maxImagesPerLabel The number of image files per label for training.
     * @param trainPartitionPercentage  The percentage of image files for training.
     *
     * @return  The array of InputSplit objects, [0] for train, [1] for test.
     * */
    public static InputSplit[] splitImageDataToTrainAndTest(
            final Random rand,
            final File imageDataPath,
            final ParentPathLabelGenerator labelMaker,
            final int maxImagesPerLabel,
            final double trainPartitionPercentage) {

        final FileSplit fileSplit = new FileSplit(
                imageDataPath,
                NativeImageLoader.ALLOWED_FORMATS,
                rand);

        final int totalLabels =
                fileSplit.getRootDir().listFiles(File::isDirectory).length;
        final int totalImages = Math.toIntExact(fileSplit.length());
        final double testPartitionPercentage = 1 - trainPartitionPercentage;

        // BalancedPathFilter random the order of datas,
        // remove data randomly to balance the total data per label for training.
        final BalancedPathFilter pathFilter = new BalancedPathFilter(
                rand, labelMaker, totalImages, totalLabels,
                maxImagesPerLabel);

        final InputSplit[] inputSplit = fileSplit.sample(
                pathFilter, trainPartitionPercentage, testPartitionPercentage);

        return inputSplit;
    }

    /**
     * Build the image transform pipeline.
     *
     * @param rand1 The Random object 1 for 1st image transform.
     * @param rand2 The Random object 2 for 2nd image transform.
     * */
    public static ImageTransform generateImageTransformPipeline(
            final Random rand1 ,
            final Random rand2,
            final double probability1,
            final double probability2,
            final boolean shuffle) {

        final ImageTransform flipTransform1 = new FlipImageTransform(rand1);
        final ImageTransform flipTransform2 = new FlipImageTransform(rand2);

        final PipelineImageTransform imageTransform = new PipelineImageTransform.Builder()
                .addImageTransform(flipTransform1, probability1)
                .addImageTransform(flipTransform2, probability2)
                .build();

        imageTransform.setShuffle(shuffle);

        return imageTransform;
    }
}
