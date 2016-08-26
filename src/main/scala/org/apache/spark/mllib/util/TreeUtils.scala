package org.apache.spark.mllib.util

import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.SparkConf
import org.apache.spark.deploy.SparkHadoopUtil


object TreeUtils {
  def getFileSystem(conf: SparkConf, path: Path): FileSystem = {
    val hadoopConf = SparkHadoopUtil.get.newConfiguration(conf)
    if (sys.env.contains("HADOOP_CONF_DIR") || sys.env.contains("YARN_CONF_DIR")) {
      val hdfsConfPath = if (sys.env.get("HADOOP_CONF_DIR").isDefined) {
        sys.env.get("HADOOP_CONF_DIR").get + "/core-site.xml"
      } else {
        sys.env.get("YARN_CONF_DIR").get + "/core-site.xml"
      }
      hadoopConf.addResource(new Path(hdfsConfPath))
    }
    path.getFileSystem(hadoopConf)
  }

  def getPartitionOffsets(upper: Int, numPartitions: Int): (Array[Int], Array[Int]) = {
    val npp = upper / numPartitions
    val nppp = npp + 1
    val residual = upper - npp * numPartitions
    val boundary = residual * nppp
    val startPP = new Array[Int](numPartitions)
    val lcLenPP = new Array[Int](numPartitions)
    var i = 0
    while(i < numPartitions) {
      if (i < residual) {
        startPP(i) = nppp * i
        lcLenPP(i) = nppp
      }
      else{
        startPP(i) = boundary + (i - residual) * npp
        lcLenPP(i) = npp
      }
      i += 1
    }
    (startPP, lcLenPP)

    /***
      * println(s"upper:$upper" + s"numPartitions: $numPartitions")
      * val kpp = {
      * val npp = upper / numPartitions
      * if (npp * numPartitions == upper) npp else npp + 1
      * }
      * val startPP = Array.tabulate(numPartitions)(_ * kpp)
      * val lcLensPP = Array.tabulate(numPartitions)(pi =>
      * if (pi < numPartitions - 1) kpp else (upper - startPP(pi))
      * )
      * (startPP, lcLensPP)**/

  }


}
