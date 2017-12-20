package pagerank;

import org.apache.spark._
import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

object PageRank {

    def main(args: Array[String]): Unit = {
        val spark = SparkSession.builder.appName(s"${this.getClass.getSimpleName}").getOrCreate()
        val sc = spark.sparkContext

        val oldGraph = GraphLoader.edgeListFile(sc, args(0))

        var graph = oldGraph.outerJoinVertices(oldGraph.outDegrees) { (id, oldAttr, outDegOpt) =>
            outDegOpt match {
                case Some(outDeg) => (1.0, outDeg)
                case None => (1.0, 0) 
            }
        }

        val alpha = 0.15
        var i = 0
        val num_iter = args(1).toInt
        for( i<- 1 to num_iter) {

            println("Iter: #" + i)

            // compute the new ranks
            val rankUpdates = graph.aggregateMessages[Double](
                triplet => {
                    triplet.sendToDst(triplet.srcAttr._1 / triplet.srcAttr._2)
                },
                (a, b) => (a + b),
                TripletFields.Src
            )

            graph = graph.outerJoinVertices(rankUpdates) { (id, oldAttr, rank) =>
                rank match {
                    case Some(r) => (alpha + (1 - alpha) * r, oldAttr._2)
                    case None => (alpha, oldAttr._2)
                }
            }

        }
        graph.vertices.collect.foreach(println(_))
    }
}
