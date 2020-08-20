import java.io.File;
import java.io.FileNotFoundException;
import java.util.Arrays;
import java.util.Scanner;

public class DataFile {
    private double[][] data;
    private int numofArg = 6;
    private static double[] maxVal;
    private double[] addr;

    public DataFile(){

    }

    public DataFile(int num){
        File testFile = new File("data.cvs");
        Scanner test = null;

        addr = new double [num];
        data = new double [num][numofArg];
        maxVal = new double[numofArg];
        Arrays.fill(maxVal,0);

        try {
            test = new Scanner(testFile);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        String line;
        String [] values;

        for(int  i=0; test.hasNextLine();i++){
            line = test.nextLine();
            values = line.split(",");
            // parse String[] to int[]
            double[] array = Arrays.stream(values).mapToDouble(Double::parseDouble).toArray();
            addr[i] = array[0];
            for(int j=1;j<7;j++){
                data[i][j-1]=array[j];
                if(array[j]>maxVal[j-1])
                    maxVal[j-1]=array[j];
            }

        }

    }

    public double[] getLine(int line){
        return data[line];
    }


    public int getAddr(int line){
        return (int)addr[line];
    }

    public double getMaxValue(int index){
        return maxVal[index];
    }
}
