import org.apache.spark.ml.clustering.LDA;
import org.apache.spark.ml.clustering.LDAModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.io.IOException;

/*
    把微博文档转换成topicDistribution
 */
public class LDATopic {
    public static void main(String[] args) {
        // Creates a SparkSession
        SparkSession spark = SparkSession
                .builder()
                .appName("LDATopic")
                .getOrCreate();

        // Loads data.
        Dataset<Row> dataset = spark.read().format("libsvm")
                .load("hdfs://hadoop-node1:9000/data/sample_lda_libsvm_data.txt");

        // Trains a LDA model.
        LDA lda = new LDA().setK(100).setMaxIter(100);
        LDAModel model = lda.fit(dataset);

        double ll = model.logLikelihood(dataset);
        double lp = model.logPerplexity(dataset);
        System.out.println("The lower bound on the log likelihood of the entire corpus: " + ll);
        System.out.println("The upper bound on perplexity: " + lp);

        // Describe topics.
        Dataset<Row> topics = model.describeTopics(10);
        System.out.println("The topics described by their top-weighted terms:");
        topics.show();

        // Shows the result.
        Dataset<Row> transformed = model.transform(dataset);
        transformed.drop("features");
        transformed.show();

        try {
            model.write().overwrite().save("hdfs://hadoop-node1:9000/model/LDAModelExample");
            transformed.write().parquet("hdfs://hadoop-node1:9000/model/docRepresentationExample.parquet");
        } catch (IOException e) {
            e.printStackTrace();
        }

        spark.stop();
    }
}
