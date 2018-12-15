package com.dl4j.server;

import com.amazonaws.auth.AWSStaticCredentialsProvider;
import com.amazonaws.auth.profile.ProfileCredentialsProvider;
import com.amazonaws.regions.Regions;
import com.amazonaws.services.rekognition.AmazonRekognition;
import com.amazonaws.services.rekognition.AmazonRekognitionClientBuilder;
import com.amazonaws.services.rekognition.model.DeleteFacesRequest;
import com.amazonaws.services.rekognition.model.FaceMatch;
import com.amazonaws.services.rekognition.model.FaceRecord;
import com.amazonaws.services.rekognition.model.Image;
import com.amazonaws.services.rekognition.model.IndexFacesRequest;
import com.amazonaws.services.rekognition.model.IndexFacesResult;
import com.amazonaws.services.rekognition.model.SearchFacesRequest;
import com.amazonaws.services.rekognition.model.SearchFacesResult;
import com.dl4j.utils.JavaCVHelper;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.ByteBuffer;
import java.util.ArrayList;
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
public class HttpWebCamRekognitionStreamServer implements Runnable {

    private static Logger log =
            LoggerFactory.getLogger(HttpWebCamRekognitionStreamServer.class);

    private static ServerSocket serverSocket;
    private static RequestHandler requestHandler;
    private static Mat mat;
    private static Thread thread;

    private final int port;
    private final FrameGrabber frameGrabber;
    private final AmazonRekognition rekognition;

    private final String boundary = "stream";

    public HttpWebCamRekognitionStreamServer(int port, int camDeviceNumber) {

        this.rekognition = AmazonRekognitionClientBuilder.standard()
                .withCredentials(new AWSStaticCredentialsProvider(
                        new ProfileCredentialsProvider().getCredentials()))
                .withRegion(Regions.US_EAST_1)
                .build();

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
            Frame frame;
            while(true) {
                try {
                    log.debug("sending image ... " + new Date());

                    frame = frameGrabber.grab();
                    JavaCVHelper.saveFrameToJPG(frame, "current-frame.jpg");

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

                    // TODO: What do with Rek
                    final IndexFacesResult indexFacesResult =
                            rekognition.indexFaces(new IndexFacesRequest()
                                    .withCollectionId("aws-sin1")
                                    .withImage(new Image()
                                            .withBytes(ByteBuffer.wrap(imageBytes))));
                    final ArrayList<FaceRecord> faceRecords =
                            new ArrayList<>(indexFacesResult.getFaceRecords());
                    String faceId;
                    for (FaceRecord faceRecord : faceRecords) {
                        faceId = faceRecord.getFace().getFaceId();
                        final SearchFacesResult searchFacesResult =
                                rekognition.searchFaces(new SearchFacesRequest()
                                        .withCollectionId("aws-sin1")
                                        .withFaceId(faceId)
                                        .withMaxFaces(1)
                                        .withFaceMatchThreshold(
                                                new Float(90)));
                        final ArrayList<FaceMatch> faceMatches =
                                new ArrayList<>(searchFacesResult.getFaceMatches());
                        for (FaceMatch faceMatch : faceMatches) {
                            System.out.println(
                                    faceId + ", "
                                            + faceMatch.getFace().getExternalImageId() + ", "
                                            + faceMatch.getFace().getConfidence()
                            );
                        }
                        rekognition.deleteFaces(new DeleteFacesRequest()
                                .withFaceIds(faceId)
                                .withCollectionId("aws-sin1"));
                    }
                    Thread.sleep(1000);
                } catch(Exception e) {
                    log.error("Error streaming image", e);
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
