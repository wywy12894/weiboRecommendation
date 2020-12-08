import org.apache.spark.ml.linalg.BLAS;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.stat.Summarizer;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;
import org.apache.spark.sql.types.DataTypes;

import static org.apache.spark.sql.functions.col;

public class ContentRecommendation {
    public static void main(String[] args) {
        // Creates a SparkSession
        SparkSession spark = SparkSession
                .builder()
                .appName("ContentRecommendation")
                .getOrCreate();

        // TopicDistribution
        String filepath3 = "hdfs://hadoop-node1:9000/model/docRepresentationExample.parquet";
        // input
        String filepath4 = "hdfs://hadoop-node1:9000/data/input.txt";
        // output
        String filepath5 = "hdfs://hadoop-node1:9000/data/output.json";
        // Chinese
        String filepath7 = "hdfs://hadoop-node1:9000/model/rootContent.parquet";

        // Load document vector
        Dataset<Row> documents = spark.read().parquet(filepath3);
        // Load input
        Dataset<Row> input = spark.read().format("libsvm").load(filepath4);
        // input vector
        input = documents.join(input,"label").drop("features");
        // other documents
        documents = documents.except(input).join(input.groupBy().agg(Summarizer.mean(col("topicDistribution"))));
        // cosine distance
        spark.udf().register("cos_func",
                (Vector v1, Vector v2)-> BLAS.dot(v1, v2)/(Math.sqrt(BLAS.dot(v1,v1))*Math.sqrt(BLAS.dot(v2,v2))),
                DataTypes.DoubleType);
        documents = documents.withColumn("cosine",
                functions.callUDF("cos_func", col("topicDistribution"), col("mean(topicDistribution)")));
        documents.show();
        // sort by cosine distance
        Dataset<Row> output = documents.sort(col("cosine").desc()).limit(10);
        output = output.drop("topicDistribution").drop("mean(topicDistribution)");
        output.show(false);

        // Load original weibo content
        Dataset<Row> rootcontent = spark.read().parquet(filepath7);
        output = output.join(rootcontent, "label");
        output.show();
        output.write().json(filepath5);


        spark.stop();
    }
}