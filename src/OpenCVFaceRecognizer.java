import java.io.File;
import java.io.FilenameFilter; 
import java.nio.IntBuffer; 
import static org.bytedeco.javacpp.opencv_face.*; 
import static org.bytedeco.javacpp.opencv_core.*; 
import static org.bytedeco.javacpp.opencv_imgcodecs.*; 
import static org.bytedeco.javacpp.opencv_imgcodecs.CV_LOAD_IMAGE_GRAYSCALE;

import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.opencv_face.FaceRecognizer;
	
public class OpenCVFaceRecognizer {
    public static void main(String[] args) {
        String trainingDir = args[0];
        Mat testImage = imread(args[1], CV_LOAD_IMAGE_GRAYSCALE);
        
        //Resize Test Image
//        Mat resizedTestImage = new Mat();
//        org.bytedeco.javacpp.opencv_core.Size newTestImageSize = new org.bytedeco.javacpp.opencv_core.Size(1000,707); 
//        org.bytedeco.javacpp.opencv_imgproc.resize(testImage, resizedTestImage, newTestImageSize);
//        testImage = null;
//        testImage = resizedTestImage;        

        File root = new File(trainingDir);

        FilenameFilter imgFilter = new FilenameFilter() {
            public boolean accept(File dir, String name) {
                name = name.toLowerCase();
                return name.endsWith(".jpg") || name.endsWith(".pgm") || name.endsWith(".png");
            }
        };

        File[] imageFiles = root.listFiles(imgFilter);

        MatVector images = new MatVector(imageFiles.length);

        Mat labels = new Mat(imageFiles.length, 1, CV_32SC1);
        IntBuffer labelsBuf = labels.createBuffer();

        int counter = 0;

        for (File image : imageFiles) {
            Mat img = imread(image.getAbsolutePath(), CV_LOAD_IMAGE_GRAYSCALE);
            int label = Integer.parseInt(image.getName().split("\\-")[0]);
            images.put(counter, img);
            labelsBuf.put(counter, label);
            counter++;
        }

  //      FaceRecognizer faceRecognizer = createFisherFaceRecognizer();
          
        // Algoritmos testados	
     //    FaceRecognizer faceRecognizer = createEigenFaceRecognizer();
        FaceRecognizer faceRecognizer = createLBPHFaceRecognizer(); //funciona  quando a foto tem resolução maior

        faceRecognizer.train(images, labels);

        IntPointer label = new IntPointer(1);
        DoublePointer confidence = new DoublePointer(1);
        faceRecognizer.predict(testImage, label, confidence);
        int predictedLabel = label.get(0);
        
        switch(predictedLabel){
        case 1:
        	System.out.println("id: "+ predictedLabel+ " Ruiva");
        	break;
        case 2:
        	System.out.println("id: "+ predictedLabel+ " Ator do Titanic");
        	break;
        case 3:
    		System.out.println("id: "+ predictedLabel+ " Loira");
    		break;
        case 4:
    		System.out.println("id: "+ predictedLabel+" Will Smith");
    		break;
        case 5:
    		System.out.println("id: "+ predictedLabel+ " Daniel Gonçalves da Silva" );
    		break;     		
        }
      
        
        
    }
}