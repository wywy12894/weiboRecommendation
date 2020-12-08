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

        String filepath8 = "/usr/project/data/rootcontent.txt";
 //       String filepath8 = "/usr/project/data/root_content.txt";
        String filepath7 = "/usr/project/model/rootContent.parquet";

        JavaRDD<Weibo> weiboRDD = spark.read()
                .textFile(filepath8)
                .javaRDD()
                .map(line -> {
                    String[] parts = line.split("\t");
                    Weibo weibo = new Weibo();
                    weibo.setLabel(Double.parseDouble(parts[0].trim()));
                    if (parts.length >= 2)
                        weibo.setConetnt(parts[1].trim());
                    return weibo;
                });

        // Apply a schema to an RDD of JavaBeans to get a DataFrame
        Dataset<Row> weiboDF = spark.createDataFrame(weiboRDD, Weibo.class);
        weiboDF.show();
        weiboDF.write().parquet(filepath7);

        spark.stop();
    }
}
