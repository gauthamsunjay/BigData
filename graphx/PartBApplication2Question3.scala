package graphx;

// imports
import org.apache.spark._
import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession;

object PartC {
    def main(args: Array[String]): Unit = {
        val spark = SparkSession.builder.appName(s"${this.getClass.getSimpleName}").getOrCreate()
        val sc = spark.sparkContext
        val vertices: RDD[(VertexId, Array[String])] = sc.textFile(args(0)).map(line => (line.split(" ")(0).toLong, line.split(" ")(1).split(",")))
        val edges: RDD[Edge[Int]] = sc.textFile(args(1)).map(line => Edge(line.split(" ")(0).toLong, line.split(" ")(1).toLong, 1))

        var graph = Graph(vertices, edges)
 
        val sumWords: VertexRDD[(Int, Int)] = graph.aggregateMessages[(Int, Int)](
          triplet => {
            triplet.sendToDst(1, triplet.srcAttr.length)
          },
          (a, b) => (a._1 + b._1, a._2 + b._2) // Reduce Function
        )

        val avgWords: VertexRDD[Double] =
          sumWords.mapValues( (id, value) =>
            value match { case (count, totalWords) => totalWords / count.toDouble } )
    
        avgWords.collect.foreach(x => println("Vertex " + x._1 + " has an average of " + x._2 + " words in its neighborhood"))
    }
}
