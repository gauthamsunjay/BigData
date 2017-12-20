package graphx;

// imports
import org.apache.spark._
import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession;

object PartA {
    def main(args: Array[String]): Unit = {
        val spark = SparkSession.builder.appName(s"${this.getClass.getSimpleName}").getOrCreate()
        val sc = spark.sparkContext
        val vertices: RDD[(VertexId, Array[String])] = sc.textFile(args(0)).map(line => (line.split(" ")(0).toLong, line.split(" ")(1).split(",")))
        val edges: RDD[Edge[Int]] = sc.textFile(args(1)).map(line => Edge(line.split(" ")(0).toLong, line.split(" ")(1).toLong, 1))

        var graph = Graph(vertices, edges)
        val strictlyLargerEdges = graph.triplets.filter(triplet => triplet.srcAttr.length > triplet.dstAttr.length).count
        println(strictlyLargerEdges)
    }
}
