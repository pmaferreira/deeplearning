import org.nd4j.linalg.dataset.api.iterator.BaseDatasetIterator;

import java.io.IOException;

public class MyIterator  extends BaseDatasetIterator {

    public MyIterator(int batch, int numExamples) throws IOException {
        super(batch, numExamples, new MyFetcher(numExamples));
    }

}
