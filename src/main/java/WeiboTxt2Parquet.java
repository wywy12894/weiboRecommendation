import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.io.Serializable;

public class WeiboTxt2Parquet {
    public static class Weibo implements Serializable {
        private double label;
        private String content;

        public Double getLabel() {
            return label;
        }

        public void setLabel(Double label) {
            this.label = label;
        }

        public String getContent() {
            return content;
        }

        public void setConetnt(String content) {
            this.content = content;
        }
    }

    public static void main(String[] args){
        // Creates a SparkSession
        SparkSession spark = SparkSession
                .builder()
                .appName("WeiboTxt2Parquet")
                .getOrCreate();

        JavaRDD<Weibo> weiboRDD = spark.read()
                .textFile("hdfs://hadoop-node1:9000/data/rootcontent.txt")
                .javaRDD()
                .map(line -> {
                    String[] parts = line.split("\t");
                    Weibo weibo = new Weibo();
                    weibo.setLabel(Double.parseDouble(parts[0].trim()));
                    weibo.setConetnt(parts[1].trim());
                    return weibo;
                });

        // Apply a schema to an RDD of JavaBeans to get a DataFrame
        Dataset<Row> peopleDF = spark.createDataFrame(weiboRDD, Weibo.class);
        peopleDF.show();
        peopleDF.write().parquet("hdfs://hadoop-node1:9000/model/rootContent.parquet");

        spark.stop();
    }
}
