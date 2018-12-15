package com.dl4j.utils;

import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.WindowConstants;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.javacv.OpenCVFrameGrabber;

public class WebCamStreamHelper implements Runnable {

    private int camDeviceNumber;
    private int imageWidth;
    private int imageHeight;
    private Frame frame;
    private Mat mat;
    private boolean loop;
    private String windowTitle;

    public WebCamStreamHelper(int camDeviceNumber, int imageWidth, int imageHeight, String windowTitle) {
        this.camDeviceNumber = camDeviceNumber;
        this.imageWidth = imageWidth;
        this.imageHeight = imageHeight;
        this.windowTitle = windowTitle;
        this.loop = true;
    }

    private void start() throws Exception {
        final JFrame window = new JFrame();
        final ImageIcon image = new ImageIcon();
        final JLabel label = new JLabel();

        final FrameGrabber frameGrabber =
                new OpenCVFrameGrabber(camDeviceNumber);

        frameGrabber.start();

        label.setIcon(image);
        window.getContentPane().add(label);
        window.setTitle(windowTitle);
        window.setResizable(true);
        window.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);

        while(loop) {
            frame = frameGrabber.grab();
            mat = new OpenCVFrameConverter.ToMat().convert(frame);
            image.setImage(JavaCVHelper.saveMatToBufferImage(mat, imageWidth, imageHeight));
            window.pack();
            label.updateUI();
            window.setVisible(true);
        }
    }

    public void run() {
        try {
            start();
        } catch(Exception e) {
            close();
        }
    }

    public void close() {
        this.loop = false;
    }

    public int getImageWidth() {
        return imageWidth;
    }

    public void setImageWidth(int imageWidth) {
        this.imageWidth = imageWidth;
    }

    public int getImageHeight() {
        return imageHeight;
    }

    public void setImageHeight(int imageHeight) {
        this.imageHeight = imageHeight;
    }

    public Frame getFrame() {
        return frame;
    }

    public void setFrame(Frame frame) {
        this.frame = frame;
    }

    public Mat getMat() {
        return mat;
    }

    public void setMat(Mat mat) {
        this.mat = mat;
    }
}
