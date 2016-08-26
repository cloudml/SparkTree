package org.apache.spark.mllib.dataSet

import breeze.collection.mutable.SparseArray
import breeze.linalg.SparseVector
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.tree.model.SplitInfo
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import scala.collection.mutable

/**
  * Created by jlinleung on 2016/5/1.
  */
class dataSet(label: Array[Short] = null,
              score: Array[Double] = null,
              queryBoundy: Array[Int] = null,
              data: RDD[(Int, SparseArray[Short], Array[SplitInfo])] = null,
              dataTransposed: RDD[(Int, org.apache.spark.mllib.linalg.Vector)] = null) {
  def getData(): RDD[(Int, SparseArray[Short], Array[SplitInfo])] = {
    data
  }

  def getDataTransposed(): RDD[(Int, org.apache.spark.mllib.linalg.Vector)] = {
    dataTransposed
  }

  def getLabel(): Array[Short] = {
    label
  }

  def getScore(): Array[Double] = {
    score
  }

  def getQueryBoundy(): Array[Int] = {
    queryBoundy
  }
}


object dataSetLoader {

  def loadData(sc: SparkContext, path: String, minPartitions: Int)
  : RDD[(Int, SparseVector[Short], Array[SplitInfo])] = {
    sc.textFile(path, minPartitions).map { line =>
      val parts = line.split("#")
      val feat = parts(0).toInt
      val samples = parts(1).split(',').map(_.toShort)

      var is = 0
      var nnz = 0
      while (is < samples.length) {
        if (samples(is) != 0) {
          nnz += 1
        }
        is += 1
      }
      val idx = new Array[Int](nnz)
      val vas = new Array[Short](nnz)
      is = 0
      nnz = 0
      while (is < samples.length) {
        if (samples(is) != 0) {
          idx(nnz) = is
          vas(nnz) = samples(is)
          nnz += 1
        }
        is += 1
      }
      val sparseSamples = new SparseVector[Short](idx, vas, nnz, is)

      val splits = if (parts.length > 2) {
        parts(2).split(',').map(threshold => new SplitInfo(feat, threshold.toDouble))
      } else {
        val maxFeat = sparseSamples.valuesIterator.max + 1
        Array.tabulate(maxFeat)(threshold => new SplitInfo(feat, threshold))
      }
      (feat, sparseSamples, splits)
    }.persist(StorageLevel.MEMORY_AND_DISK).setName("trainingData")
  }

  def loadTrainingDataForLambdamart(sc: SparkContext, path: String, minPartitions: Int,
                                    sampleQueryId: Array[Int], QueryBound: Array[Int], numSampling: Int)
  : RDD[(Int, SparseArray[Short], Array[SplitInfo])] = {
    var rdd = sc.textFile(path, minPartitions).map { line =>
      val parts = line.split("#")
      val feat = parts(0).toInt
      val samples = parts(1).split(',').map(_.toShort)
      var is = 0
      // sampling data
      val samplingData = if (sampleQueryId == null) samples
      else {
        val sd = new Array[Short](numSampling)
        var it = 0
        var icur = 0
        while (it < sampleQueryId.length) {
          val query = sampleQueryId(it)
          for (is <- QueryBound(query) until QueryBound(query + 1)) {
            sd(icur) = samples(is)
            icur += 1
          }
          it += 1
        }
        sd
      }

      //      // Sparse data
      //      is = 0
      //      var nnz = 0
      //      while (is < samplingData.length) {
      //        if (samplingData(is) != 0) {
      //          nnz += 1
      //        }
      //        is += 1
      //      }
      //      val idx = new Array[Int](nnz)
      //      val vas = new Array[Short](nnz)
      //      is = 0
      //      nnz = 0
      //      while (is < samplingData.length) {
      //        if (samplingData(is) != 0) {
      //          idx(nnz) = is
      //          vas(nnz) = samplingData(is)
      //          nnz += 1
      //        }
      //        is += 1
      //      }
      //      val sparseSamples = new SparseVector[Short](idx, vas, nnz, is)

      val v2no = new mutable.HashMap[Short, Int]().withDefaultValue(0)
      is = 0
      while (is < samplingData.length) {
        v2no(samplingData(is)) += 1
        is += 1
      }
      val (default, numDefault) = v2no.maxBy(x => x._2)
      val numAct = samplingData.length - numDefault
      val idx = new Array[Int](numAct)
      val vas = new Array[Short](numAct)
      is = 0
      var nnz = 0
      while (is < samplingData.length) {
        if (samplingData(is) != default) {
          idx(nnz) = is
          vas(nnz) = samplingData(is)
          nnz += 1
        }
        is += 1
      }
      val sparseSamples = new SparseArray[Short](idx, vas, nnz, is, default)

      //      val sparseSamples = new SparseVector[Short](sparseArr)


      val splits = if (parts.length > 2) {
        parts(2).split(',').map(threshold => new SplitInfo(feat, threshold.toDouble))
      } else {
        val maxFeat = samples.max + 1
        Array.tabulate(maxFeat)(threshold => new SplitInfo(feat, threshold))
      }
      (feat, sparseSamples, splits)
    }
    rdd = sc.getConf.getOption("lambdaMart_numPartitions").map(_.toInt) match {
      case Some(np) =>
        println("repartitioning")
        rdd.sortBy(x => x._1, numPartitions = np)

      case None => rdd
    }

    rdd.persist(StorageLevel.MEMORY_AND_DISK).setName("trainingData")
  }

  def loadTrainingDataForClassification(sc: SparkContext, path: String, minPartitions: Int, numDoc: Int)
  : RDD[(Int, SparseArray[Short], Array[SplitInfo])] = {
    var rdd = sc.textFile(path, minPartitions).map { line =>
      val parts = line.split("#")
      val feat = parts(0).toInt
      val samples = parts(1).split(',').map(_.toShort)
      var is = 0
      // sampling data
      val samplingData = if (numDoc == samples.length) {
        samples
      }
      else if(numDoc<samples.length){
        samples.take(numDoc)
      }
      else {
        val arr = new Array[Short](numDoc)
        while(is<numDoc){
          arr(is)=samples(is%samples.length)
          is+=1
        }
        arr
      }

      val v2no = new mutable.HashMap[Short, Int]().withDefaultValue(0)
      is = 0
      while (is < samplingData.length) {
        v2no(samplingData(is)) += 1
        is += 1
      }
      val (default, numDefault) = v2no.maxBy(x => x._2)
      val numAct = samplingData.length - numDefault
      val idx = new Array[Int](numAct)
      val vas = new Array[Short](numAct)
      is = 0
      var nnz = 0
      while (is < samplingData.length) {
        if (samplingData(is) != default) {
          idx(nnz) = is
          vas(nnz) = samplingData(is)
          nnz += 1
        }
        is += 1
      }
      val sparseSamples = new SparseArray[Short](idx, vas, nnz, numDoc, default)

      val splits = if (parts.length > 2) {
        parts(2).split(',').map(threshold => new SplitInfo(feat, threshold.toDouble))
      } else {
        val maxFeat = samples.max + 1
        Array.tabulate(maxFeat)(threshold => new SplitInfo(feat, threshold))
      }
      (feat, sparseSamples, splits)
    }

    rdd = sc.getConf.getOption("lambdaMart_numPartitions").map(_.toInt) match {
      case Some(np) =>
        println("repartitioning")
        rdd.sortBy(x => x._1, numPartitions = np)

      case None => rdd
    }

    rdd.persist(StorageLevel.MEMORY_AND_DISK).setName("trainingData")
  }

  def loadDataTransposed(sc: SparkContext, path: String): RDD[(Int, org.apache.spark.mllib.linalg.Vector)] = {
    sc.textFile(path).map { line =>
      val parts = line.split("#")
      val sId = parts(0).toInt
      val features = parts(1).split(",").map(_.toDouble)
      (sId, Vectors.dense(features))
    }.persist(StorageLevel.MEMORY_AND_DISK).setName("validationData")
  }

  def loadlabelScores(sc: SparkContext, path: String): Array[Short] = {
    sc.textFile(path).first().split(',').map(_.toShort)
  }

  def loadInitScores(sc: SparkContext, path: String): Array[Double] = {
    sc.textFile(path).first().split(',').map(_.toDouble)
  }

  def loadQueryBoundy(sc: SparkContext, path: String): Array[Int] = {

    sc.textFile(path).first().split(',').map(_.toInt)
  }

  def loadThresholdMap(sc: SparkContext, path: String, numFeats: Int): Array[Array[Double]] = {
    val thresholdMapTuples = sc.textFile(path).map { line =>
      val fields = line.split("#", 2)
      (fields(0).toInt, fields(1).split(',').map(_.toDouble))
    }.collect()
    val numFeatsTM = thresholdMapTuples.length
    assert(numFeats == numFeatsTM, s"ThresholdMap file contains $numFeatsTM features that != $numFeats")
    val thresholdMap = new Array[Array[Double]](numFeats)
    thresholdMapTuples.foreach { case (fi, thresholds) =>
      thresholdMap(fi) = thresholds
    }
    thresholdMap
  }

  def loadFeature2NameMap(sc: SparkContext, path: String): Array[String] = {
    sc.textFile(path).map(line => line.split("#")(1)).collect()
  }

  def getSampleLabels(testQueryId: Array[Int], QueryBound: Array[Int], labels: Array[Short]): Array[Short] = {
    println("parse test labels")
    val testLabels = new Array[Short](labels.length)
    var it = 0
    var icur = 0
    while (it < testQueryId.length) {
      val query = testQueryId(it)
      for (is <- QueryBound(query) until QueryBound(query + 1)) {
        testLabels(icur) = labels(is)
        icur += 1
      }
      it += 1
    }
    testLabels.dropRight(labels.length - icur)
  }

  def getSampleInitScores(trainingQueryId: Array[Int], QueryBound: Array[Int], scores: Array[Double], len: Int): Array[Double] = {
    println("parse init scores")
    val trainingScores = new Array[Double](len)
    var it = 0
    var icur = 0
    while (it < trainingQueryId.length) {
      val query = trainingQueryId(it)
      for (is <- QueryBound(query) until QueryBound(query + 1)) {
        trainingScores(icur) = scores(is)
        icur += 1
      }
      it += 1
    }
    trainingScores
  }

  def getSampleQueryBound(QueryId: Array[Int], queryBoundy: Array[Int]): Array[Int] = {
    println("get query bound")
    val res = new Array[Int](QueryId.length + 1)
    res(0) = 0
    var qid = 0
    while (qid < QueryId.length) {
      res(qid + 1) = res(qid) + queryBoundy(QueryId(qid) + 1) - queryBoundy(QueryId(qid))
      qid += 1
    }
    res
  }

  def getSampleFeatureData(sc: SparkContext, trainingData: RDD[(Int, SparseArray[Short], Array[SplitInfo])], sampleFeatPct: Double) = {
    //    def IsSeleted(ffraction: Double): Boolean = {
    //      val randomNum = scala.util.Random.nextDouble()
    //      var active = false
    //      if(randomNum < ffraction)
    //      {
    //        active = true
    //      }
    //      active
    //    }
    val rdd = if (sampleFeatPct < 1.0) {
      var sampleData = trainingData.sample(false, sampleFeatPct)
      //      sampleData = sc.getConf.getOption("lambdaMart_numPartitions").map(_.toInt) match {
      //        case Some(np) => sampleData.sortBy(x => x._1, numPartitions = np)
      //        case None => sampleData
      //      }
      //      val sampleData = trainingData.filter(item =>IsSeleted(sampleFeatPct))
      val numFeats_S = sampleData.count()
      println(s"numFeats_sampling: $numFeats_S")
      println(s"numPartitions_sampling: ${sampleData.partitions.length}")
      trainingData.unpersist(blocking = false)
      sampleData
    } else trainingData
    rdd.persist(StorageLevel.MEMORY_AND_DISK).setName("sampleTrainingData")
  }
}
