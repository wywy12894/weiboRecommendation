import org.apache.spark.ml.clustering.LDA;
import org.apache.spark.ml.clustering.LDAModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.io.IOException;

public class LDATopic {

    public static void main(String[] args) {
        // Creates a SparkSession
        SparkSession spark = SparkSession
                .builder()
                .appName("LDATopic")
                .getOrCreate();

        // contentWithNum
//        String filepath1 = "hdfs://hadoop-node1:9000/data/contentWithNum.txt";
        String filepath1 = "/usr/project/data/content.txt";
        // LDAModel
        String filepath2 = "/usr/project/model/LDAModelExample";
        // TopicDistribution
        String filepath3 = "/usr/project/model/docRepresentationExample.parquet";

        // Loads data.
        Dataset<Row> dataset = spark.read().format("libsvm")
                .load(filepath1);

        // Trains a LDA model.
        LDA lda = new LDA().setK(50).setMaxIter(100);
        LDAModel model = lda.fit(dataset);

        double ll = model.logLikelihood(dataset);
        double lp = model.logPerplexity(dataset);
        System.out.println("The lower bound on the log likelihood of the entire corpus: " + ll);
        System.out.println("The upper bound on perplexity: " + lp);

        // Describe topics.
        Dataset<Row> topics = model.describeTopics(5);
        System.out.println("The topics described by their top-weighted terms:");
        topics.show(false);

        // Shows the result.
        Dataset<Row> transformed = model.transform(dataset).drop("features");
        transformed.show();

        try {
            model.write().overwrite().save(filepath2);
            transformed.write().parquet(filepath3);
        } catch (IOException e) {
            e.printStackTrace();
        }

        spark.stop();
    }
}
