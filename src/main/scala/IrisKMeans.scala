import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.{SparkConf, SparkContext}

import scala.io.Source

object IrisKMeans {
  def main(args:Array[String]) {
    println("First spark-mllib example")

    val appName = "IrisKMeans"
    val master = "local"
    val conf = new SparkConf().setAppName(appName).setMaster(master)
    val sc = new SparkContext(conf)

    println("loading iris data from URL")

    val url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    val src = Source.fromURL(url).getLines().filter(_.size>0).toList
    val textData = sc.parallelize(src)
    val parsedData = textData
      .map(_.split(",").dropRight(1).map(_.toDouble))
        .map(Vectors.dense(_)).cache()

    val numClusters = 3
    val numIterations =20
    val clusters = KMeans.train(parsedData, numClusters, numIterations)

    val WSSSE = clusters.computeCost(parsedData)
    println("Within set sum of squared errors = " + WSSSE)


  }
}
