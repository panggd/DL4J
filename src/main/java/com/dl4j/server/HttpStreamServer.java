package com.dl4j.server;

import com.dl4j.utils.JavaCVHelper;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.Java2DFrameConverter;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.Date;

/**
 * https://github.com/mesutpiskin/opencv-livestream-over-http
 * */
public class HttpStreamServer implements Runnable {

    private static Logger log =
            LoggerFactory.getLogger(HttpStreamServer.class);

    private static ServerSocket serverSocket;
    private static RequestHandler requestHandler;
    private static Thread thread;
    private int port;
    private final String boundary = "stream";
    public Mat frame;

    public HttpStreamServer(int port, Mat frame) {
        this.port = port;
        this.frame = frame;
    }

    public void run() {
        try {
            log.debug("Stream server running on localhost:" + port);
            startStreamingServer();
            while(true) {
                requestHandler = new RequestHandler(serverSocket.accept(), frame);
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
        private Mat frame;
        private OutputStream outputStream;

        public RequestHandler(Socket clientSocket, Mat frame)
                throws IOException {

            this.clientSocket = clientSocket;
            this.frame = frame;

            writeHeader(this.clientSocket.getOutputStream(), boundary);
        }

        public void run() {
            boolean loop = true;
            while(loop) {
                if(frame == null) {
                    log.debug("frame is null");
                    return;
                }

                try {
                    log.debug("sending image ... " + new Date());
                    outputStream = clientSocket.getOutputStream();
                    BufferedImage img = JavaCVHelper.saveMatToBufferImage(frame);
                    ByteArrayOutputStream baos = new ByteArrayOutputStream();
                    ImageIO.write(img, "jpg", baos);
                    byte[] imageBytes = baos.toByteArray();
                    outputStream.write(("Content-type: image/jpeg\r\n" +
                            "Content-Length: " + imageBytes.length + "\r\n" +
                            "\r\n").getBytes());
                    outputStream.write(imageBytes);
                    outputStream.write(("\r\n--" + boundary + "\r\n").getBytes());
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
                    "Cache-Control: no-store, no-cache, must-revalidate, pre-check=0, post-check=0, max-age=0\r\n" +
                    "Pragma: no-cache\r\n" +
                    "Content-Type: multipart/x-mixed-replace; " +
                    "boundary=" + boundary + "\r\n" +
                    "\r\n" +
                    "--" + boundary + "\r\n").getBytes());
        }
    }
}
