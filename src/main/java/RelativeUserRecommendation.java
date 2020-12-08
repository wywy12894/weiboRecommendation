import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.graphx.Edge;
import org.apache.spark.graphx.Graph;
import org.apache.spark.graphx.lib.LabelPropagation;
import org.apache.spark.graphx.lib.PageRank;
import org.apache.spark.rdd.RDD;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.storage.StorageLevel;
import scala.Tuple2;
import scala.reflect.ClassTag;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class RelativeUserRecommendation {
    public static void main(String[] args) throws IOException {
        if (args.length < 1) {
            System.err.println("input user");
            System.exit(1);
        }
        // Creates a SparkSession
        SparkSession spark = SparkSession
                .builder()
                .appName("RelativeUserRecommendation")
                .getOrCreate();

        String filepath6 = "/usr/project/data/followers.txt";
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

        Graph<Object,String> result2 = LabelPropagation.run(followGraph, 5, stringTag);


        System.out.println("=====================================");

        long query = Long.parseLong(args[0]);
        long querylabel = (long)result2.vertices().toJavaRDD().filter(tuple->(long)tuple._1() == query).first()._2;
        JavaRDD<Edge<String>> edges = edgeJavaRDD.filter(edge->edge.srcId() == query);
        Set<Long> neighbors = new HashSet<>();
        neighbors.add(query);
        edges.foreach(edge-> neighbors.add(edge.dstId()));
        List<Tuple2<Object, Object>> res = result2.vertices().toJavaRDD()
                .filter(tuple->(long)tuple._2() == querylabel)
                .filter(tuple-> ! neighbors.contains((long)tuple._1())).collect();
        System.out.println(res);
        System.out.println("=====================================");
        BufferedWriter bw = new BufferedWriter(new FileWriter("/usr/project/output/RelativeUserRecommendation.txt"));
        bw.write(res.toString());
        bw.close();

        spark.stop();
    }
}
