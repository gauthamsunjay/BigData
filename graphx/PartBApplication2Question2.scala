package graphx;

// imports
import org.apache.spark._
import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession;

object PartB {
    def main(args: Array[String]): Unit = {
        
        val spark = SparkSession.builder.appName(s"${this.getClass.getSimpleName}").getOrCreate()
        val sc = spark.sparkContext
        val vertices: RDD[(VertexId, Array[String])] = sc.textFile(args(0)).map(line => (line.split(" ")(0).toLong, line.split(" ")(1).split(",")))
        val edges: RDD[Edge[Int]] = sc.textFile(args(1)).map(line => Edge(line.split(" ")(0).toLong, line.split(" ")(1).toLong, 1))

        var graph = Graph(vertices, edges)
        
        // Define a reduce operation to compute the highest degree vertex
        def max(a: (VertexId, (Int, Int)), b: (VertexId, (Int, Int))): (VertexId, (Int, Int)) = {
          if (a._2._2 > b._2._2) a
          else if(b._2._2 > a._2._2) b 
          else {
            if (a._2._1 > b._2._1) a else b
          }  
        }

        val degreeGraph = graph.outerJoinVertices(graph.outDegrees) { (id, oldAttr, outDegOpt) =>
          outDegOpt match {
            case Some(outDeg) => (oldAttr.length,outDeg)
            case None => (oldAttr.length,0) // No outDegree means zero outDegree
          }
        }

        println(degreeGraph.vertices.reduce(max)._1)
    }
}
