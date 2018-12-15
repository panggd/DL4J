package com.dl4j.javacv;

import com.dl4j.server.HttpWebCamRekognitionStreamServer;

/**
 * This main program show how we start a webcam stream using JavaCV.
 * Useful when we want to do some real time ML inference.
 * */
public class WebCamStream {
    public static void main(String[] args) {
//        HttpWebCamS3StreamServer httpWebCamS3StreamServer =
//                new HttpWebCamS3StreamServer(7000, 0);
//        new Thread(httpWebCamS3StreamServer).start();

        HttpWebCamRekognitionStreamServer httpWebCamRekognitionStreamServer =
                new HttpWebCamRekognitionStreamServer(7000, 0);
        new Thread(httpWebCamRekognitionStreamServer).start();
    }
}
