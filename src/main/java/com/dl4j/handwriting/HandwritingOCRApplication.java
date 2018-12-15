package com.dl4j.handwriting;

import com.amazonaws.auth.AWSStaticCredentialsProvider;
import com.amazonaws.auth.profile.ProfileCredentialsProvider;
import com.amazonaws.regions.Regions;
import com.amazonaws.services.rekognition.AmazonRekognition;
import com.amazonaws.services.rekognition.AmazonRekognitionClientBuilder;
import com.amazonaws.services.rekognition.model.DetectTextRequest;
import com.amazonaws.services.rekognition.model.Image;
import com.amazonaws.services.rekognition.model.TextDetection;
import com.amazonaws.util.IOUtils;
import com.dl4j.utils.HandwritingExtractor;
import com.dl4j.utils.JavaCVHelper;
import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import javax.swing.BorderFactory;
import javax.swing.BoxLayout;
import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JSplitPane;
import javax.swing.JTextArea;
import javax.swing.JTextField;
import javax.swing.border.Border;
import javax.swing.event.DocumentEvent;
import javax.swing.event.DocumentListener;
import javax.swing.filechooser.FileSystemView;
import org.bytedeco.javacpp.opencv_core;

import static org.bytedeco.javacpp.opencv_imgcodecs.imread;

/**
 * An GUI version of HandwritingOCR.java
 * */
public class HandwritingOCRApplication {

    private int lineImageWidthVal = 800;
    private int lineImageHeightVal = 600;
    private int lineKernelWidthVal = 100;
    private int lineKernelHeightVal = 300;
    private int lineThresholdVal = 75;
    private int lineContourAreaLimitVal = 2000;
    private int lineContourHeightLimitVal = 20;
    private int lineContourWidthVal = 800;
    private int lineContourHeightVal = 150;

    private int wordImageWidthVal = 800;
    private int wordImageHeightVal = 100;
    private int wordKernelWidthVal = 28;
    private int wordKernelHeightVal = 32;
    private int wordThresholdVal = 85;
    private int wordContourAreaLimitVal = 50;
    private int wordContourHeightLimitVal = 20;
    private int wordContourWidthVal = 800;
    private int wordContourHeightVal = 150;

    private String filePath;
    private String output;

    private AmazonRekognition rekognition;

    private JTextArea ocrOutput;

    public void run() {
        rekognition =
                AmazonRekognitionClientBuilder.standard()
                        .withCredentials(new AWSStaticCredentialsProvider(
                                new ProfileCredentialsProvider().getCredentials()))
                        .withRegion(Regions.DEFAULT_REGION)
                        .build();

        output = "";
        ocrOutput = new JTextArea(output,10, 20);

        final JFrame jFrame = new JFrame();
        final JLabel label = new JLabel();
        int textFieldWidth = 5;

        JFileChooser jFileChooser =
                new JFileChooser(
                        FileSystemView.getFileSystemView()
                                .getHomeDirectory());

        JButton inputButton = new JButton("Open image file");
        inputButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                int returnValue = jFileChooser.showOpenDialog(null);
                if (returnValue == JFileChooser.APPROVE_OPTION) {
                    filePath = jFileChooser.getSelectedFile()
                            .getAbsolutePath();
                    try {
                        opencv_core.Mat mat = imread(filePath);
                        BufferedImage img = JavaCVHelper.saveMatToBufferImage(
                                mat, lineImageWidthVal, lineImageHeightVal);
                        label.setIcon(new ImageIcon(img));
                    } catch(IOException ioe) {
                        ioe.printStackTrace();
                    }
                }
            }
        });

        JTextField imageWidth = new JTextField(
                lineImageWidthVal + "", textFieldWidth);
        imageWidth.getDocument().addDocumentListener(new DocumentListener() {
            public void changedUpdate(DocumentEvent e) {
                update();
            }
            public void removeUpdate(DocumentEvent e) {
                update();
            }
            public void insertUpdate(DocumentEvent e) {
                update();
            }
            private void update() {
                try {
                    lineImageWidthVal = Integer.parseInt(
                            imageWidth.getText());
                } catch(NumberFormatException nfe) {
                    nfe.printStackTrace();
                }
            }
        });

        JTextField imageHeight = new JTextField(
                lineImageHeightVal + "", textFieldWidth);
        imageHeight.getDocument().addDocumentListener(new DocumentListener() {
            public void changedUpdate(DocumentEvent e) {
                update();
            }
            public void removeUpdate(DocumentEvent e) {
                update();
            }
            public void insertUpdate(DocumentEvent e) {
                update();
            }
            private void update() {
                try {
                    lineImageHeightVal = Integer.parseInt(
                            imageHeight.getText());
                } catch(NumberFormatException nfe) {
                    nfe.printStackTrace();
                }
            }
        });

        JTextField kernelWidth = new JTextField(
                lineKernelWidthVal + "", textFieldWidth);
        kernelWidth.getDocument().addDocumentListener(new DocumentListener() {
            public void changedUpdate(DocumentEvent e) {
                update();
            }
            public void removeUpdate(DocumentEvent e) {
                update();
            }
            public void insertUpdate(DocumentEvent e) {
                update();
            }
            private void update() {
                try {
                    lineKernelWidthVal = Integer.parseInt(
                            kernelWidth.getText());
                } catch(NumberFormatException nfe) {
                    nfe.printStackTrace();
                }
            }
        });

        JTextField kernelHeight = new JTextField(
                lineKernelHeightVal + "", textFieldWidth);
        kernelHeight.getDocument().addDocumentListener(new DocumentListener() {
            public void changedUpdate(DocumentEvent e) {
                update();
            }
            public void removeUpdate(DocumentEvent e) {
                update();
            }
            public void insertUpdate(DocumentEvent e) {
                update();
            }
            private void update() {
                try {
                    lineKernelHeightVal = Integer.parseInt(
                            kernelHeight.getText());
                } catch(NumberFormatException nfe) {
                    nfe.printStackTrace();
                }
            }
        });

        JTextField threshold = new JTextField(
                lineThresholdVal + "", textFieldWidth);
        threshold.getDocument().addDocumentListener(new DocumentListener() {
            public void changedUpdate(DocumentEvent e) {
                update();
            }
            public void removeUpdate(DocumentEvent e) {
                update();
            }
            public void insertUpdate(DocumentEvent e) {
                update();
            }
            private void update() {
                try {
                    lineThresholdVal = Integer.parseInt(
                            threshold.getText());
                } catch(NumberFormatException nfe) {
                    nfe.printStackTrace();
                }
            }
        });

        JTextField contourAreaLimit = new JTextField(
                lineContourAreaLimitVal + "", textFieldWidth);
        contourAreaLimit.getDocument().addDocumentListener(new DocumentListener() {
            public void changedUpdate(DocumentEvent e) {
                update();
            }
            public void removeUpdate(DocumentEvent e) {
                update();
            }
            public void insertUpdate(DocumentEvent e) {
                update();
            }
            private void update() {
                try {
                    lineContourAreaLimitVal = Integer.parseInt(
                            contourAreaLimit.getText());
                } catch(NumberFormatException nfe) {
                    nfe.printStackTrace();
                }
            }
        });

        JTextField contourHeightLimit = new JTextField(
                lineContourHeightLimitVal + "", textFieldWidth);
        contourHeightLimit.getDocument().addDocumentListener(new DocumentListener() {
            public void changedUpdate(DocumentEvent e) {
                update();
            }
            public void removeUpdate(DocumentEvent e) {
                update();
            }
            public void insertUpdate(DocumentEvent e) {
                update();
            }
            private void update() {
                try {
                    lineContourHeightLimitVal = Integer.parseInt(
                            contourHeightLimit.getText());
                } catch(NumberFormatException nfe) {
                    nfe.printStackTrace();
                }
            }
        });

        JTextField contourWidth = new JTextField(
                lineContourWidthVal + "", textFieldWidth);
        contourWidth.getDocument().addDocumentListener(new DocumentListener() {
            public void changedUpdate(DocumentEvent e) {
                update();
            }
            public void removeUpdate(DocumentEvent e) {
                update();
            }
            public void insertUpdate(DocumentEvent e) {
                update();
            }
            private void update() {
                try {
                    lineContourWidthVal = Integer.parseInt(
                            contourWidth.getText());
                } catch(NumberFormatException nfe) {
                    nfe.printStackTrace();
                }
            }
        });

        JTextField contourHeight = new JTextField(
                lineContourHeightVal + "", textFieldWidth);
        contourHeight.getDocument().addDocumentListener(new DocumentListener() {
            public void changedUpdate(DocumentEvent e) {
                update();
            }
            public void removeUpdate(DocumentEvent e) {
                update();
            }
            public void insertUpdate(DocumentEvent e) {
                update();
            }
            private void update() {
                try {
                    lineContourHeightVal = Integer.parseInt(
                            contourHeight.getText());
                } catch(NumberFormatException nfe) {
                    nfe.printStackTrace();
                }
            }
        });

        JTextField wordImageWidth = new JTextField(
                wordImageWidthVal + "", textFieldWidth);
        wordImageWidth.getDocument().addDocumentListener(new DocumentListener() {
            public void changedUpdate(DocumentEvent e) {
                update();
            }
            public void removeUpdate(DocumentEvent e) {
                update();
            }
            public void insertUpdate(DocumentEvent e) {
                update();
            }
            private void update() {
                try {
                    wordImageWidthVal = Integer.parseInt(
                            wordImageWidth.getText());
                } catch(NumberFormatException nfe) {
                    nfe.printStackTrace();
                }
            }
        });

        JTextField wordImageHeight = new JTextField(
                wordImageHeightVal + "", textFieldWidth);
        wordImageHeight.getDocument().addDocumentListener(new DocumentListener() {
            public void changedUpdate(DocumentEvent e) {
                update();
            }
            public void removeUpdate(DocumentEvent e) {
                update();
            }
            public void insertUpdate(DocumentEvent e) {
                update();
            }
            private void update() {
                try {
                    wordImageHeightVal = Integer.parseInt(
                            wordImageHeight.getText());
                } catch(NumberFormatException nfe) {
                    nfe.printStackTrace();
                }
            }
        });

        JTextField wordKernelWidth = new JTextField(
                wordKernelWidthVal + "", textFieldWidth);
        wordKernelWidth.getDocument().addDocumentListener(new DocumentListener() {
            public void changedUpdate(DocumentEvent e) {
                update();
            }
            public void removeUpdate(DocumentEvent e) {
                update();
            }
            public void insertUpdate(DocumentEvent e) {
                update();
            }
            private void update() {
                try {
                    wordKernelWidthVal = Integer.parseInt(
                            wordKernelWidth.getText());
                } catch(NumberFormatException nfe) {
                    nfe.printStackTrace();
                }
            }
        });

        JTextField wordKernelHeight = new JTextField(
                wordKernelHeightVal + "", textFieldWidth);
        wordKernelHeight.getDocument().addDocumentListener(new DocumentListener() {
            public void changedUpdate(DocumentEvent e) {
                update();
            }
            public void removeUpdate(DocumentEvent e) {
                update();
            }
            public void insertUpdate(DocumentEvent e) {
                update();
            }
            private void update() {
                try {
                    wordKernelHeightVal = Integer.parseInt(
                            wordKernelHeight.getText());
                } catch(NumberFormatException nfe) {
                    nfe.printStackTrace();
                }
            }
        });

        JTextField wordThreshold = new JTextField(
                wordThresholdVal + "", textFieldWidth);
        wordThreshold.getDocument().addDocumentListener(new DocumentListener() {
            public void changedUpdate(DocumentEvent e) {
                update();
            }
            public void removeUpdate(DocumentEvent e) {
                update();
            }
            public void insertUpdate(DocumentEvent e) {
                update();
            }
            private void update() {
                try {
                    wordThresholdVal = Integer.parseInt(
                            wordThreshold.getText());
                } catch(NumberFormatException nfe) {
                    nfe.printStackTrace();
                }
            }
        });

        JTextField wordContourAreaLimit = new JTextField(
                wordContourAreaLimitVal + "", textFieldWidth);
        imageWidth.getDocument().addDocumentListener(new DocumentListener() {
            public void changedUpdate(DocumentEvent e) {
                update();
            }
            public void removeUpdate(DocumentEvent e) {
                update();
            }
            public void insertUpdate(DocumentEvent e) {
                update();
            }
            private void update() {
                try {
                    wordContourAreaLimitVal = Integer.parseInt(
                            wordContourAreaLimit.getText());
                } catch(NumberFormatException nfe) {
                    nfe.printStackTrace();
                }
            }
        });

        JTextField wordContourHeightLimit = new JTextField(
                wordContourHeightLimitVal + "", textFieldWidth);
        wordContourHeightLimit.getDocument().addDocumentListener(new DocumentListener() {
            public void changedUpdate(DocumentEvent e) {
                update();
            }
            public void removeUpdate(DocumentEvent e) {
                update();
            }
            public void insertUpdate(DocumentEvent e) {
                update();
            }
            private void update() {
                try {
                    wordContourHeightLimitVal = Integer.parseInt(
                            wordContourHeightLimit.getText());
                } catch(NumberFormatException nfe) {
                    nfe.printStackTrace();
                }
            }
        });

        JTextField wordContourWidth = new JTextField(
                wordContourWidthVal + "", textFieldWidth);
        wordContourWidth.getDocument().addDocumentListener(new DocumentListener() {
            public void changedUpdate(DocumentEvent e) {
                update();
            }
            public void removeUpdate(DocumentEvent e) {
                update();
            }
            public void insertUpdate(DocumentEvent e) {
                update();
            }
            private void update() {
                try {
                    wordContourWidthVal = Integer.parseInt(
                            wordContourWidth.getText());
                } catch(NumberFormatException nfe) {
                    nfe.printStackTrace();
                }
            }
        });

        JTextField wordContourHeight = new JTextField(
                wordContourHeightVal + "", textFieldWidth);
        wordContourHeight.getDocument().addDocumentListener(new DocumentListener() {
            public void changedUpdate(DocumentEvent e) {
                update();
            }
            public void removeUpdate(DocumentEvent e) {
                update();
            }
            public void insertUpdate(DocumentEvent e) {
                update();
            }
            private void update() {
                try {
                    wordContourHeightVal = Integer.parseInt(
                            wordContourHeight.getText());
                } catch(NumberFormatException nfe) {
                    nfe.printStackTrace();
                }
            }
        });

        Border emptyBorder = BorderFactory.createEmptyBorder(0, 0, 10, 0);

        JPanel lineImageWidthPanel = new JPanel(new BorderLayout());
        lineImageWidthPanel.add(new JLabel("Image Width"), BorderLayout.NORTH);
        lineImageWidthPanel.add(imageWidth, BorderLayout.CENTER);
        lineImageWidthPanel.setBorder(emptyBorder);

        JPanel lineImageHeightPanel = new JPanel(new BorderLayout());
        lineImageHeightPanel.add(new JLabel("Image Height"), BorderLayout.NORTH);
        lineImageHeightPanel.add(imageHeight, BorderLayout.CENTER);
        lineImageHeightPanel.setBorder(emptyBorder);

        JPanel lineKernelWidthPanel = new JPanel(new BorderLayout());
        lineKernelWidthPanel.add(new JLabel("Kernel Width"), BorderLayout.NORTH);
        lineKernelWidthPanel.add(kernelWidth, BorderLayout.CENTER);
        lineKernelWidthPanel.setBorder(emptyBorder);

        JPanel lineKernelHeightPanel = new JPanel(new BorderLayout());
        lineKernelHeightPanel.add(new JLabel("Kernel Height"), BorderLayout.NORTH);
        lineKernelHeightPanel.add(kernelHeight, BorderLayout.CENTER);
        lineKernelHeightPanel.setBorder(emptyBorder);

        JPanel lineThresholdPanel = new JPanel(new BorderLayout());
        lineThresholdPanel.add(new JLabel("Threshold"), BorderLayout.NORTH);
        lineThresholdPanel.add(threshold, BorderLayout.CENTER);
        lineThresholdPanel.setBorder(emptyBorder);

        JPanel lineContourAreaLimitPanel = new JPanel(new BorderLayout());
        lineContourAreaLimitPanel.add(new JLabel("Contour Area Limit"), BorderLayout.NORTH);
        lineContourAreaLimitPanel.add(contourAreaLimit, BorderLayout.CENTER);
        lineContourAreaLimitPanel.setBorder(emptyBorder);

        JPanel lineContourHeightLimitPanel = new JPanel(new BorderLayout());
        lineContourHeightLimitPanel.add(new JLabel("Contour Height Limit"), BorderLayout.NORTH);
        lineContourHeightLimitPanel.add(contourHeightLimit, BorderLayout.CENTER);
        lineContourHeightLimitPanel.setBorder(emptyBorder);

        JPanel lineContourWidthPanel = new JPanel(new BorderLayout());
        lineContourWidthPanel.add(new JLabel("Contour Width"), BorderLayout.NORTH);
        lineContourWidthPanel.add(contourWidth, BorderLayout.CENTER);
        lineContourWidthPanel.setBorder(emptyBorder);

        JPanel lineContourHeightPanel = new JPanel(new BorderLayout());
        lineContourHeightPanel.add(new JLabel("Contour Height"), BorderLayout.NORTH);
        lineContourHeightPanel.add(contourHeight, BorderLayout.CENTER);
        lineContourHeightPanel.setBorder(emptyBorder);

        JPanel wordImageWidthPanel = new JPanel(new BorderLayout());
        wordImageWidthPanel.add(new JLabel("Image Width"), BorderLayout.NORTH);
        wordImageWidthPanel.add(wordImageWidth, BorderLayout.CENTER);
        wordImageWidthPanel.setBorder(emptyBorder);

        JPanel wordImageHeightPanel = new JPanel(new BorderLayout());
        wordImageHeightPanel.add(new JLabel("Image Height"), BorderLayout.NORTH);
        wordImageHeightPanel.add(wordImageHeight, BorderLayout.CENTER);
        wordImageHeightPanel.setBorder(emptyBorder);

        JPanel wordKernelWidthPanel = new JPanel(new BorderLayout());
        wordKernelWidthPanel.add(new JLabel("Kernel Width"), BorderLayout.NORTH);
        wordKernelWidthPanel.add(wordKernelWidth, BorderLayout.CENTER);
        wordKernelWidthPanel.setBorder(emptyBorder);

        JPanel wordKernelHeightPanel = new JPanel(new BorderLayout());
        wordKernelHeightPanel.add(new JLabel("Kernel Height"), BorderLayout.NORTH);
        wordKernelHeightPanel.add(wordKernelHeight, BorderLayout.CENTER);
        wordKernelHeightPanel.setBorder(emptyBorder);

        JPanel wordThresholdPanel = new JPanel(new BorderLayout());
        wordThresholdPanel.add(new JLabel("Threshold"), BorderLayout.NORTH);
        wordThresholdPanel.add(wordThreshold, BorderLayout.CENTER);
        wordThresholdPanel.setBorder(emptyBorder);

        JPanel wordContourAreaLimitPanel = new JPanel(new BorderLayout());
        wordContourAreaLimitPanel.add(new JLabel("Contour Area Limit"), BorderLayout.NORTH);
        wordContourAreaLimitPanel.add(wordContourAreaLimit, BorderLayout.CENTER);
        wordContourAreaLimitPanel.setBorder(emptyBorder);

        JPanel wordContourHeightLimitPanel = new JPanel(new BorderLayout());
        wordContourHeightLimitPanel.add(new JLabel("Contour Height Limit"), BorderLayout.NORTH);
        wordContourHeightLimitPanel.add(wordContourHeightLimit, BorderLayout.CENTER);
        wordContourHeightLimitPanel.setBorder(emptyBorder);

        JPanel wordContourWidthPanel = new JPanel(new BorderLayout());
        wordContourWidthPanel.add(new JLabel("Contour Width"), BorderLayout.NORTH);
        wordContourWidthPanel.add(wordContourWidth, BorderLayout.CENTER);
        wordContourWidthPanel.setBorder(emptyBorder);

        JPanel wordContourHeightPanel = new JPanel(new BorderLayout());
        wordContourHeightPanel.add(new JLabel("Contour Height"), BorderLayout.NORTH);
        wordContourHeightPanel.add(wordContourHeight, BorderLayout.CENTER);
        wordContourHeightPanel.setBorder(emptyBorder);

        JPanel inputPanel = new JPanel(new BorderLayout());
        inputPanel.add(inputButton, BorderLayout.CENTER);

        JPanel imagePanel = new JPanel(new BorderLayout());
        imagePanel.add(label, BorderLayout.CENTER);

        imagePanel.add(new JScrollPane(ocrOutput), BorderLayout.SOUTH);

        JPanel configPanel = new JPanel();
        configPanel.setLayout(new BoxLayout(configPanel, BoxLayout.Y_AXIS));
        configPanel.add(new JLabel("Sentence Parameters"));
        configPanel.add(lineImageWidthPanel);
        configPanel.add(lineImageHeightPanel);
        configPanel.add(lineKernelWidthPanel);
        configPanel.add(lineKernelHeightPanel);
        configPanel.add(lineThresholdPanel);
        configPanel.add(lineContourAreaLimitPanel);
        configPanel.add(lineContourHeightLimitPanel);
        configPanel.add(lineContourWidthPanel);
        configPanel.add(lineContourHeightPanel);

        configPanel.add(new JLabel("Word Parameters"));
        configPanel.add(wordImageWidthPanel);
        configPanel.add(wordImageHeightPanel);
        configPanel.add(wordKernelWidthPanel);
        configPanel.add(wordKernelHeightPanel);
        configPanel.add(wordThresholdPanel);
        configPanel.add(wordContourAreaLimitPanel);
        configPanel.add(wordContourHeightLimitPanel);
        configPanel.add(wordContourWidthPanel);
        configPanel.add(wordContourHeightPanel);

        JSplitPane centerPanel =
                new JSplitPane(
                        JSplitPane.HORIZONTAL_SPLIT,
                        imagePanel,
                        new JScrollPane(configPanel));

        centerPanel.setResizeWeight(0.9);
        centerPanel.setBorder(BorderFactory.createLineBorder(Color.BLACK));

        JButton rekognitionButton = new JButton("Perform OCR");
        rekognitionButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                try {
                    output = "";
                    Arrays.stream(new File(System.getProperty("user.dir"))
                            .listFiles((f, p) -> p.endsWith("jpg")))
                            .forEach(File::delete);
                    Arrays.stream(new File(System.getProperty("user.dir"))
                            .listFiles((f, p) -> p.endsWith("log")))
                            .forEach(File::delete);
                    performOCR();
                } catch(Exception ex) {
                    ex.printStackTrace();
                }
            }
        });

        JPanel rekognitionPanel = new JPanel(new BorderLayout());
        rekognitionPanel.add(rekognitionButton, BorderLayout.CENTER);

        JPanel mainPanel = new JPanel(new BorderLayout());
        mainPanel.add(inputPanel, BorderLayout.NORTH);
        mainPanel.add(centerPanel, BorderLayout.CENTER);
        mainPanel.add(rekognitionPanel, BorderLayout.SOUTH);

        jFrame.setLayout(new BorderLayout());
        jFrame.add(mainPanel, BorderLayout.CENTER);

        jFrame.setSize(new Dimension(1200, 1000));
        jFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        jFrame.setLocationRelativeTo(null);
        jFrame.setTitle("Handwriting OCR");
        jFrame.setVisible(true);
    }

    private void performOCR() throws Exception {

        final HandwritingExtractor lineExtractor =
                new HandwritingExtractor(
                        filePath);

        lineExtractor.setImageWidth(lineImageWidthVal);
        lineExtractor.setImageHeight(lineImageHeightVal);
        lineExtractor.setKernelWidth(lineKernelWidthVal);
        lineExtractor.setKernelHeight(lineKernelHeightVal);
        lineExtractor.setThreshold(lineThresholdVal);
        lineExtractor.setContourAreaLimit(lineContourAreaLimitVal);
        lineExtractor.setContourHeightLimit(lineContourHeightLimitVal);
        lineExtractor.setContourWidth(lineContourWidthVal);
        lineExtractor.setContourHeight(lineContourHeightVal);
        lineExtractor.setSaveContour(true);
        lineExtractor.setShowWindow(false);
        lineExtractor.setContourPrefix("line");
        lineExtractor.setWindowTitle("Sentences");

        // Step 1: Extract the sentence to image files
        lineExtractor.extract();
        lineExtractor.saveAsImageFile("lines.jpg");

        final ArrayList<String> extractList = lineExtractor.getExtractList();

//        Collections.reverse(extractList);

        // Step 2: Extract word to image files
        HandwritingExtractor wordExtractor;
        for(int i=0; i<extractList.size(); i++) {

            String extractFile = extractList.get(i);

            String fileName = "line-" + i + "-word";

            wordExtractor = new HandwritingExtractor(extractFile);
            wordExtractor.setImageWidth(wordImageWidthVal);
            wordExtractor.setImageHeight(wordImageHeightVal);
            wordExtractor.setKernelWidth(wordKernelWidthVal);
            wordExtractor.setKernelHeight(wordKernelHeightVal);
            wordExtractor.setThreshold(wordThresholdVal);
            wordExtractor.setContourAreaLimit(wordContourAreaLimitVal);
            wordExtractor.setContourHeightLimit(wordContourHeightLimitVal);
            wordExtractor.setContourWidth(wordContourWidthVal);
            wordExtractor.setContourHeight(wordContourHeightVal);
            wordExtractor.setSaveContour(true);
            wordExtractor.setContourPrefix(fileName);
            wordExtractor.setWindowTitle(extractFile);

            wordExtractor.extract();
            wordExtractor.saveAsImageFile(fileName + ".jpg");

            byte[] byteArray = IOUtils.toByteArray(
                    new FileInputStream(new File(fileName + ".jpg")));

            Image image = new Image()
                    .withBytes(ByteBuffer.wrap(byteArray));

            ArrayList<TextDetection> textDetections =
                    (ArrayList<TextDetection>)
                            rekognition.detectText(new DetectTextRequest()
                                    .withImage(image)
                            ).getTextDetections();

            for(TextDetection textDetection: textDetections) {
                output += textDetection.getDetectedText() +
                        ", " + textDetection.getType() + "\n";
            }
            ocrOutput.setText(output);
        }
    }
}
