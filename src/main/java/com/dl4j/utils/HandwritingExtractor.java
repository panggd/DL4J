package com.dl4j.utils;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.WindowConstants;

import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.MatVector;
import org.bytedeco.javacpp.opencv_core.Rect;
import org.bytedeco.javacpp.opencv_core.Scalar;
import org.bytedeco.javacpp.opencv_core.Size;
import org.bytedeco.javacpp.opencv_core.Point;

import static org.bytedeco.javacpp.opencv_core.CV_8UC1;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;
import static org.bytedeco.javacpp.opencv_imgcodecs.imwrite;
import static org.bytedeco.javacpp.opencv_imgproc.boundingRect;
import static org.bytedeco.javacpp.opencv_imgproc.contourArea;
import static org.bytedeco.javacpp.opencv_imgproc.cvtColor;
import static org.bytedeco.javacpp.opencv_imgproc.dilate;
import static org.bytedeco.javacpp.opencv_imgproc.findContours;
import static org.bytedeco.javacpp.opencv_imgproc.rectangle;
import static org.bytedeco.javacpp.opencv_imgproc.resize;
import static org.bytedeco.javacpp.opencv_imgproc.threshold;
import static org.bytedeco.javacpp.opencv_imgproc.CV_AA;
import static org.bytedeco.javacpp.opencv_imgproc.COLOR_BGR2GRAY;
import static org.bytedeco.javacpp.opencv_imgproc.THRESH_BINARY_INV;
import static org.bytedeco.javacpp.opencv_imgproc.CV_RETR_EXTERNAL;
import static org.bytedeco.javacpp.opencv_imgproc.CV_CHAIN_APPROX_SIMPLE;

/**
 * This class pre-process an image with handwritten text,
 * perform contours, bbox and finally crop the bound areas to image files.
 *
 * It has the flexibility to fine tune parameters for pre-processing.
 * */
public class HandwritingExtractor {

    private String imagePath;
    private String windowTitle;
    private int imageWidth;
    private int imageHeight;
    private int kernelWidth;
    private int kernelHeight;
    private int threshold;
    private int contourAreaLimit;
    private int contourHeightLimit;
    private int contourWidth;
    private int contourHeight;
    private boolean saveContour;
    private boolean showWindow;
    private String contourPrefix;
    private Mat mat;
    private ArrayList<String> extractList;

    public HandwritingExtractor(String imagePath) {
        this.imagePath = imagePath;
        this.kernelWidth = 25;
        this.kernelHeight = 25;
        this.threshold = 155;
        this.contourAreaLimit = 50;
        this.contourHeightLimit = 28;
        this.windowTitle = "Handwriting Extractor";
        this.extractList = new ArrayList<>();
        this.saveContour = true;
        this.showWindow = false;
        this.contourPrefix = "";
    }

    public void extract() throws Exception {

        final JFrame window = new JFrame();
        final ImageIcon image = new ImageIcon();
        final JLabel label = new JLabel();

        final Mat grey = new Mat();
        final Mat thres = new Mat();
        final Mat dist = new Mat();
        final MatVector contours = new MatVector();
        final Mat kernel = Mat.ones(kernelWidth, kernelHeight, CV_8UC1).asMat();

        mat = imread(imagePath);

        // pre-process the image before finding contours.
        cvtColor(mat, grey, COLOR_BGR2GRAY);
        threshold(grey, thres, threshold, 255, THRESH_BINARY_INV);
        dilate(thres, dist, kernel);

        findContours(dist, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

        Mat contour;
        Rect rect;
        ArrayList<Rect> contourRectangles = new ArrayList();
        for(int i=0; i<contours.size(); i++) {

            contour = contours.get(i);
            double area = contourArea(contour);
            rect = boundingRect(contour);

            if(area > contourAreaLimit && rect.height() > contourHeightLimit) {
                contourRectangles.add(rect);
            }
        }

        // Sort rect on x axis
        Collections.sort(contourRectangles, new Comparator<Rect>() {
            @Override
            public int compare(Rect o1, Rect o2) {
                int result = Integer.compare(o1.x(), o2.x());
                return result;
            }
        });

        String fileName;
        for(int i=0; i<contourRectangles.size(); i++) {
            Rect rec = contourRectangles.get(i);
            if(saveContour) {
                fileName = contourPrefix + "-00" + i + ".jpg";
                Mat write = new Mat();
                resize(new Mat(mat, rec), write, new Size(contourWidth, contourHeight));
                imwrite(fileName, write);
                extractList.add(fileName);
            }
        }

        for(int i=0; i<contourRectangles.size(); i++) {
            Rect rec = contourRectangles.get(i);
            rectangle(mat,
                    new Point(rec.x(), rec.y()),
                    new Point(rec.x() + rec.width(), rec.y() + rec.height()),
                    Scalar.GREEN, 3, CV_AA, 0);
        }

        label.setIcon(image);
        window.getContentPane().add(label);
        window.setTitle(windowTitle);
        window.setResizable(true);
        window.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);

        image.setImage(JavaCVHelper.saveMatToBufferImage(
                mat, imageWidth, imageHeight));
        window.pack();
        label.updateUI();
        window.setVisible(showWindow);
    }

    public void saveAsImageFile(String fileName) {
        JavaCVHelper.saveMatToJPG(mat, imageWidth, imageHeight, fileName);
    }

    public void setContourAreaLimit(int contourAreaLimit) {
        this.contourAreaLimit = contourAreaLimit;
    }

    public void setContourHeightLimit(int contourHeightLimit) {
        this.contourHeightLimit = contourHeightLimit;
    }

    public void setContourWidth(int contourWidth) {
        this.contourWidth = contourWidth;
    }

    public void setContourHeight(int contourHeight) {
        this.contourHeight = contourHeight;
    }

    public void setSaveContour(boolean saveContour) {
        this.saveContour = saveContour;
    }

    public void setShowWindow(boolean showWindow) {
        this.showWindow = showWindow;
    }

    public void setContourPrefix(String contourPrefix) {
        this.contourPrefix = contourPrefix;
    }

    public void setWindowTitle(String windowTitle) {
        this.windowTitle = windowTitle;
    }

    public void setImageWidth(int imageWidth) {
        this.imageWidth = imageWidth;
    }

    public void setImageHeight(int imageHeight) {
        this.imageHeight = imageHeight;
    }

    public void setKernelWidth(int kernelWidth) {
        this.kernelWidth = kernelWidth;
    }

    public void setKernelHeight(int kernelHeight) {
        this.kernelHeight = kernelHeight;
    }

    public void setThreshold(int threshold) {
        this.threshold = threshold;
    }

    public ArrayList<String> getExtractList() {
        return extractList;
    }
}
