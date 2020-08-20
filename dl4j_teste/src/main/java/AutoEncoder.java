import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.BaseDatasetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.AdaGrad;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.*;
import java.util.List;



/**Example: Anomaly Detection on MNIST using simple autoencoder without pretraining
 * The goal is to identify outliers digits, i.e., those digits that are unusual or
 * not like the typical digits.
 * This is accomplished in this example by using reconstruction error: stereotypical
 * examples should have low reconstruction error, whereas outliers should have high
 * reconstruction error. The number of epochs here is set to 3. Set to 30 for even better
 * results.
 *
 * @author Alex Black
 */
public class AutoEncoder {

    public static boolean visualize = true;

    public static void main(String[] args) throws Exception {

        //Set up network. 784 in/out (as MNIST images are 28x28).
        //784 -> 250 -> 10 -> 250 -> 784
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .weightInit(WeightInit.XAVIER)
                .updater(new AdaGrad(0.05))
                .activation(Activation.RELU)
                .l2(0.0001)
                .list()
                .layer(new DenseLayer.Builder().nIn(6).nOut(5)
                        .build())
                .layer(new DenseLayer.Builder().nIn(5).nOut(3)
                        .build())
                .layer(new DenseLayer.Builder().nIn(3).nOut(5)
                        .build())
                .layer(new OutputLayer.Builder().nIn(5).nOut(6)
                        .lossFunction(LossFunctions.LossFunction.MSE)
                        .build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.setListeners(Collections.singletonList(new ScoreIterationListener(10)));

        //Load data and split into training and testing sets. 40000 train, 10000 test
        //DataSetIterator iter = new MnistDataSetIterator(100,50000,false);
        //RecordReader rr = new CSVRecordReader();
        //rr.initialize(new FileSplit(new File("data.cvs")));
        //DataSetIterator iter = new RecordReaderDataSetIterator(rr, 30,0,6,true);
        DataSetIterator iter = new MyIterator(100,3000);


        List<INDArray> featuresTrain = new ArrayList<>();
        List<INDArray> featuresTest = new ArrayList<>();
        List<INDArray> labelsTest = new ArrayList<>();

        Random r = new Random(10);
        while(iter.hasNext()){
            DataSet ds = iter.next();
            SplitTestAndTrain split = ds.splitTestAndTrain(80, r);  //80/20 split (from miniBatch = 100)
            featuresTrain.add(split.getTrain().getFeatures());
            DataSet dsTest = split.getTest();
            featuresTest.add(dsTest.getFeatures());
            INDArray indexes = Nd4j.argMax(dsTest.getLabels(),1); //Convert from one-hot representation -> index
            labelsTest.add(indexes);
        }




        //Train model:
        int nEpochs = 30;
        for( int epoch=0; epoch<nEpochs; epoch++ ){
            for(INDArray data : featuresTrain){
                net.fit(data,data);
            }
            System.out.println("Epoch " + epoch + " complete");
        }


        System.out.println("model trained!!!");

        //Evaluate the model on the test data
        //Score each example in the test set separately
        //Compose a map that relates each digit to a list of (score, example) pairs
        //Then find N best and N worst scores per digit


        Map<Integer,List<Pair<Double,INDArray>>> listsByDigit = new HashMap<>();
        for( int i=0; i<=66; i++ ) listsByDigit.put(i,new ArrayList<>());

        for( int i=0; i<featuresTest.size(); i++ ){
            INDArray testData = featuresTest.get(i);
            INDArray labels = labelsTest.get(i);
            int nRows = testData.rows();
            for( int j=0; j<nRows; j++){
                INDArray example = testData.getRow(j, true);
                int digit = (int)labels.getDouble(j);

                double score = net.score(new DataSet(example,example));
                // Add (score, example) pair to the appropriate list
                List digitAllPairs = listsByDigit.get(digit);
                digitAllPairs.add(new ImmutablePair<>(score, digit));
            }
        }

        INDArray outlier = Nd4j.zeros(DataType.DOUBLE,1, 6);
        String str = "1,3600,3,0,0,2.9999166666666666";
        String [] val;
        val = str.split(",");
        double[] array = Arrays.stream(val).mapToDouble(Double::parseDouble).toArray();
        DataFile data = new DataFile();
        for (int i =0; i<outlier.length();i++){
            outlier.putScalar(i,array[i]/data.getMaxValue(i));
        }




        System.out.println("\n\nSCORES:\n");
        double max = 0;
        for(int i =0; i<listsByDigit.size();i++) {
            for(int a=0; a<listsByDigit.get(i).size();a++){
                System.out.println("addr: "+listsByDigit.get(i).get(a).getValue() + " score: "+listsByDigit.get(i).get(a).getKey());
                if(max<listsByDigit.get(i).get(a).getKey())
                    max = listsByDigit.get(i).get(a).getKey();
            }
        }

        System.out.println("\n\n\n teste:   scrore sybil: "+ net.score(new DataSet(outlier,outlier)));
        outlier = Nd4j.zeros(DataType.DOUBLE,1, 6);
        System.out.println("\n\n\n teste:   scrore0: "+ net.score(new DataSet(outlier,outlier)));

        System.out.println("\n\n\n teste:   max scrore: "+ max);


        /*

        //Sort each list in the map by score
        Comparator<Pair<Double, INDArray>> c = new Comparator<Pair<Double, INDArray>>() {
            @Override
            public int compare(Pair<Double, INDArray> o1, Pair<Double, INDArray> o2) {
                return Double.compare(o1.getLeft(),o2.getLeft());
            }
        };

        for(List<Pair<Double, INDArray>> digitAllPairs : listsByDigit.values()){
            Collections.sort(digitAllPairs, c);
        }

        //After sorting, select N best and N worst scores (by reconstruction error) for each digit, where N=5
        List<INDArray> best = new ArrayList<>(50);
        List<INDArray> worst = new ArrayList<>(50);
        for( int i=0; i<10; i++ ){
            List<Pair<Double,INDArray>> list = listsByDigit.get(i);
            for( int j=0; j<5; j++ ){
                best.add(list.get(j).getRight());
                worst.add(list.get(list.size()-j-1).getRight());
            }
        }
        */


    }


}
