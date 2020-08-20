import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.fetcher.BaseDataFetcher;
import org.nd4j.linalg.factory.Nd4j;

import java.io.BufferedInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;

public class MyFetcher extends BaseDataFetcher {

    public static int NUM_EXAMPLES = 0;
    protected int [] order;
    private double featureData[][] = null;
    private DataFile data;

    public MyFetcher(int numExamples) throws IOException {
        InputStream is = new BufferedInputStream(new FileInputStream("data.cvs"));
        try {
            byte[] c = new byte[1024];
            int count = 0;
            int readChars = 0;
            boolean empty = true;
            while ((readChars = is.read(c)) != -1) {
                empty = false;
                for (int i = 0; i < readChars; ++i) {
                    if (c[i] == '\n') {
                        ++count;
                    }
                }
            }
            NUM_EXAMPLES =  (count == 0 && !empty) ? 1 : count;
        } finally {
            is.close();
        }

        totalExamples = NUM_EXAMPLES;
        inputColumns = 6;
        cursor =0;

        this.data = new DataFile(NUM_EXAMPLES);

        numOutcomes = 2;

        order = new int [NUM_EXAMPLES];
        for(int i =0 ; i<order.length;i++)
            order[i]=i;

        reset();

    }


    @Override
    public void fetch(int numExamples) {
        if (!hasMore()) {
            throw new IllegalStateException("Unable to get more; there are no more images");
        }

        INDArray labels = Nd4j.zeros(DataType.DOUBLE, numExamples, 100);

        if(featureData == null || featureData.length < numExamples){
            featureData = new double[numExamples][6];
        }


        for (int i = 0; i < numExamples; i++, cursor++) {
            if (!hasMore())
                break;
            double [] datarow = data.getLine(i);
            int label = data.getAddr(i);

            labels.put(i,label,1.0f);

            for(int j=0;j<6;j++)
                featureData[i][j]=datarow[j];
        }

        INDArray features;

        //normalize all values
        for(int i =0;i<featureData.length;i++)
            for(int j =0; j<featureData[i].length;j++)
                featureData[i][j] = featureData[i][j]/data.getMaxValue(j);

        features = Nd4j.create(featureData);

        curr = new DataSet(features,labels);
    }

    @Override
    public void reset(){
        cursor = 0;
        curr = null;
    }



    @Override
    public DataSet next() {
        DataSet next = super.next();
        return next;
    }

    public double getMaxValue (int line){
        return data.getMaxValue(line);
    }

    }
