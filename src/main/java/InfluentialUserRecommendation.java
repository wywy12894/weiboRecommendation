import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.graphx.Edge;
import org.apache.spark.graphx.Graph;
import org.apache.spark.graphx.lib.PageRank;
import org.apache.spark.rdd.RDD;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.storage.StorageLevel;
import scala.Tuple2;
import scala.reflect.ClassTag;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.List;

public class InfluentialUserRecommendation {
    public static void main(String[] args) throws IOException {
        // Creates a SparkSession
        SparkSession spark = SparkSession
                .builder()
                .appName("InfluentialUserRecommendation")
                .getOrCreate();

        String filepath6 = "/data/followers.txt";
//        String filepath6 = "/usr/project/data/followers.txt";

        JavaRDD<Edge<String>> edgeJavaRDD = spark.read()
                .textFile(filepath6)
                .javaRDD()
                .map(line->{
                    String[] pair = line.split(" ");
                    return new Edge<>(Integer.parseInt(pair[0].trim()), Integer.parseInt(pair[1].trim()), "follow");
                });
        RDD<Edge<String>> edgeRDD = JavaRDD.toRDD(edgeJavaRDD);

        ClassTag<String> stringTag = scala.reflect.ClassTag$.MODULE$.apply(String.class);

        Graph<String,String> followGraph = Graph.fromEdges(edgeRDD, "", StorageLevel.MEMORY_ONLY(),
                StorageLevel.MEMORY_ONLY(), stringTag, stringTag);

        Graph<Object,Object> result1 = PageRank.run(followGraph, 20, 0.01, stringTag, stringTag);

        BufferedWriter bw = new BufferedWriter(new FileWriter("/output/InfluentialUserRecommendationOutput.txt"));
        System.out.println("=====================================");
        JavaRDD<Tuple2<Object, Object>> v = result1.vertices().toJavaRDD()
                .sortBy(tuple->tuple._2, false, 0);
        List<Tuple2<Object, Object>> output =v.collect();
        int index = 0;
        for (Tuple2<?,?> tuple : output) {
            if(index > 10)
                break;
            index++;
            bw.write(tuple._1().toString());
            bw.write(" has rank: ");
            bw.write(tuple._2().toString());
            bw.newLine();
//            System.out.println(tuple._1() + " has rank: " + tuple._2() + ".");
        }
        System.out.println("=====================================");
        bw.close();
        spark.stop();
    }
}
