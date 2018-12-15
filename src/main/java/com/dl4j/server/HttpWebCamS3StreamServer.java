package com.dl4j.server;

import com.amazonaws.auth.AWSStaticCredentialsProvider;
import com.amazonaws.auth.profile.ProfileCredentialsProvider;
import com.amazonaws.regions.Regions;
import com.amazonaws.services.s3.AmazonS3;
import com.amazonaws.services.s3.AmazonS3ClientBuilder;
import com.dl4j.utils.JavaCVHelper;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.OutputStream;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.Date;
import javax.imageio.ImageIO;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.FrameGrabber;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.javacv.OpenCVFrameGrabber;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * This class will stream frame every 6 sec from web cam and upload to S3.
 * */
public class HttpWebCamS3StreamServer implements Runnable {

    private static Logger log =
            LoggerFactory.getLogger(HttpWebCamS3StreamServer.class);

    private static ServerSocket serverSocket;
    private static RequestHandler requestHandler;
    private static Mat mat;
    private static Thread thread;

    private final int port;
    private final FrameGrabber frameGrabber;
    private final AmazonS3 s3Client;
    private final String s3BucketName;
    private final String s3KeyName;

    private final String boundary = "stream";

    public HttpWebCamS3StreamServer(int port, int camDeviceNumber) {

        this.s3Client = AmazonS3ClientBuilder.standard()
                .withCredentials(new AWSStaticCredentialsProvider(
                        new ProfileCredentialsProvider().getCredentials()))
                .withRegion(Regions.US_EAST_1)
                .build();
        this.s3BucketName = "<USE YOUR BUCKET>";
        this.s3KeyName = "current-frame.jpg";

        if(!s3Client.doesBucketExistV2(s3BucketName)) {
            s3Client.createBucket(s3BucketName);
        }

        this.port = port;
        this.frameGrabber = new OpenCVFrameGrabber(camDeviceNumber);
    }

    public void run() {
        try {
            log.debug("Web cam stream running on localhost:" + port);
            startStreamingServer();
            frameGrabber.start();
            while(true) {
                requestHandler = new RequestHandler(serverSocket.accept());
                thread = new Thread(requestHandler);
                thread.start();
            }
        } catch(IOException ioe) {
            log.error("Error streaming at run()", ioe);
        }
    }

    public void close() throws IOException {
        finalize();
    }

    protected void finalize() throws IOException {
        serverSocket.close();
    }

    private void startStreamingServer() throws IOException {
        this.serverSocket = new ServerSocket(port);
    }

    private class RequestHandler implements Runnable {
        private Socket clientSocket;
        private OutputStream outputStream;

        public RequestHandler(Socket clientSocket)
                throws IOException {
            this.clientSocket = clientSocket;
            writeHeader(this.clientSocket.getOutputStream(), boundary);
        }

        public void run() {
            boolean loop = true;
            Frame frame;
            while(loop) {
                try {
                    log.debug("sending image ... " + new Date());

                    frame = frameGrabber.grab();
                    JavaCVHelper.saveFrameToJPG(frame, s3KeyName);

                    mat = new OpenCVFrameConverter.ToMat().convert(frame);
                    BufferedImage img = JavaCVHelper.saveMatToBufferImage(mat);
                    ByteArrayOutputStream baos = new ByteArrayOutputStream();
                    ImageIO.write(img, "jpg", baos);
                    byte[] imageBytes = baos.toByteArray();

                    outputStream = clientSocket.getOutputStream();
                    outputStream.write(("Content-type: image/jpeg\r\n" +
                            "Content-Length: " + imageBytes.length + "\r\n" +
                            "\r\n").getBytes());
                    outputStream.write(imageBytes);
                    outputStream.write(("\r\n--" + boundary + "\r\n").getBytes());

                    s3Client.putObject(s3BucketName, s3KeyName, new File(s3KeyName));

                    Thread.sleep(3000);
                } catch(Exception e) {
                    log.error("Error streaming image", e);
                    loop = false;
                }
            }
        }

        private void writeHeader(OutputStream stream, String boundary)
                throws IOException {
            stream.write(("HTTP/1.0 200 OK\r\n" +
                    "Connection: close\r\n" +
                    "Max-Age: 0\r\n" +
                    "Expires: 0\r\n" +
                    "Cache-Control: no-store, no-cache, must-revalidate, " +
                    "pre-check=0, post-check=0, max-age=0\r\n" +
                    "Pragma: no-cache\r\n" +
                    "Content-Type: multipart/x-mixed-replace; " +
                    "boundary=" + boundary + "\r\n" +
                    "\r\n" +
                    "--" + boundary + "\r\n").getBytes());
        }
    }
}
