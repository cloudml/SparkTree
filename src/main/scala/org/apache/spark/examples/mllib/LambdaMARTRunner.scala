package org.apache.spark.examples.mllib

import breeze.collection.mutable.SparseArray
import org.apache.hadoop.fs.Path
import org.apache.spark.mllib.dataSet.{dataSet, dataSetLoader}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.tree.config.Algo
import org.apache.spark.mllib.tree.model.SplitInfo
import org.apache.spark.mllib.tree.model.ensemblemodels.GradientBoostedDecisionTreesModel
import org.apache.spark.mllib.tree.{DerivativeCalculator, LambdaMART, config}
import org.apache.spark.mllib.util.{MLUtils, TreeUtils, treeAggregatorFormat}
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{HashPartitioner, SparkConf, SparkContext}
import scopt.OptionParser

import scala.language.reflectiveCalls
import scala.util.Random


object LambdaMARTRunner {

  class Params(var trainingData: String = null,
               var queryBoundy: String = null,
               var label: String = null,
               var initScores: String = null,
               var testData: String = null,
               var testQueryBound: String = null,
               var testLabel: String = null,
               var validationData: String = null,
               var queryBoundyValidate: String = null,
               var initScoreValidate: String = null,
               var labelValidate: String = null,
               var featureNoToFriendlyName: String = null,
               var outputTreeEnsemble: String = null,
               var expandTreeEnsemble: Boolean = false,
               var featureIniFile: String = null,
               var gainTableStr: String = null,
               var algo: String = "LambdaMart",
               var learningStrategy: String = "sgd",
               var maxDepth: Array[Int] = null,
               var numLeaves: Int = 0,
               var numPruningLeaves: Array[Int] = null,
               var numIterations: Array[Int] = null,
               var maxSplits: Int = 128,
               var learningRate: Array[Double] = null,
               var minInstancesPerNode: Array[Int] = null,
               var testSpan: Int = 0,
               var sampleFeaturePercent: Double = 1.0,
               var sampleQueryPercent: Double = 1.0,
               var sampleDocPercent: Double = 1.0,
               var numPartitions: Int = 160,
               var ffraction: Double = 1.0,
               var sfraction: Double = 1.0,
               var secondaryMS: Double = 0.0,
               var secondaryLE: Boolean = false,
               var sigma: Double = 1.0,
               var distanceWeight2: Boolean = false,
               var baselineAlpha: Array[Double] = null,
               var baselineAlphaFilename: String = null,
               var entropyCoefft: Double = 0.0,
               var featureFirstUsePenalty: Double = 0.0,
               var featureReusePenalty: Double = 0.0,
               var outputNdcgFilename: String = null,
               var active_lambda_learningStrategy: Boolean = false,
               var rho_lambda: Double = 0.5,
               var active_leaves_value_learningStrategy: Boolean = false,
               var rho_leave: Double = 0.5,
               var GainNormalization: Boolean = false,
               var feature2NameFile: String = null,
               var validationSpan: Int = 10,
               var useEarlystop: Boolean = true,
               var secondGainsFileName: String = null,
               var secondaryInverseMaxDcgFileName: String = null,
               var secondGains: Array[Double] = null,
               var secondaryInverseMaxDcg: Array[Double] = null,
               var discountsFilename: String = null,
               var discounts: Array[Double] = null,
               var sampleWeightsFilename: String = null,
               var sampleWeights: Array[Double] = null,
               var baselineDcgsFilename: String = null,
               var baselineDcgs: Array[Double] = null) extends java.io.Serializable {

    override def toString: String = {
      val propertiesStr = s"trainingData = $trainingData\nqueryBoundy = $queryBoundy\nlabel = $label\ninitScores = $initScores\n" +
        s"testData = $testData\ntestQueryBound = $testQueryBound\ntestLabel = $testLabel\ntestSpan = $testSpan\n" +
        s"sampleFeaturePercent = $sampleFeaturePercent\nsampleQueryPercent = $sampleQueryPercent\nsampleDocPercent = $sampleDocPercent\n" +
        s"outputTreeEnsemble = $outputTreeEnsemble\nfeatureNoToFriendlyName = $featureNoToFriendlyName\nvalidationData = $validationData\n" +
        s"queryBoundyValidate = $queryBoundyValidate\ninitScoreValidate = $initScoreValidate\nlabelValidate = $labelValidate\n" +
        s"expandTreeEnsemble = $expandTreeEnsemble\nfeatureIniFile = $featureIniFile\ngainTableStr = $gainTableStr\n" +
        s"algo = $algo\nmaxDepth = ${maxDepth.mkString(":")}\nnumLeaves = $numLeaves\nnumPruningLeaves = ${numPruningLeaves.mkString(":")}\nnumIterations = ${numIterations.mkString(":")}\nmaxSplits = $maxSplits\n" +
        s"learningRate = ${learningRate.mkString(":")}\nminInstancesPerNode = ${minInstancesPerNode.mkString(":")}\nffraction = $ffraction\nsfraction = $sfraction\n"


      propertiesStr
    }
  }

  def main(args: Array[String]) {
    val defaultParams = new Params()

    val parser = new OptionParser[Unit]("LambdaMART") {
      head("LambdaMART: an implementation of LambdaMART for FastRank.")

      opt[String]("trainingData") required() foreach { x =>
        defaultParams.trainingData = x
      } text ("trainingData path")
      opt[String]("queryBoundy") optional() foreach { x =>
        defaultParams.queryBoundy = x
      } text ("queryBoundy path")
      opt[String]("label") required() foreach { x =>
        defaultParams.label = x
      } text ("label path to training dataset")
      opt[String]("initScores") optional() foreach { x =>
        defaultParams.initScores = x
      } text (s"initScores path to training dataset. If not given, initScores will be {0 ...}.")

      opt[String]("testData") optional() foreach { x =>
        defaultParams.testData = x
      } text ("testData path")
      opt[String]("testQueryBound") optional() foreach { x =>
        defaultParams.testQueryBound = x
      } text ("test queryBoundy path")
      opt[String]("testLabel") optional() foreach { x =>
        defaultParams.testLabel = x
      } text ("label path to test dataset")

      opt[String]("vd") optional() foreach { x =>
        defaultParams.validationData = x
      } text ("validationData path")
      opt[String]("qbv") optional() foreach { x =>
        defaultParams.queryBoundyValidate = x
      } text ("path to queryBoundy for validation data")
      opt[String]("lv") optional() foreach { x =>
        defaultParams.labelValidate = x
      } text ("path to label for validation data")
      opt[String]("isv") optional() foreach { x =>
        defaultParams.initScoreValidate = x
      } text (s"path to initScore for validation data. If not given, initScores will be {0 ...}.")

      opt[String]("outputTreeEnsemble") required() foreach { x =>
        defaultParams.outputTreeEnsemble = x
      } text ("outputTreeEnsemble path")
      opt[String]("ftfn") optional() foreach { x =>
        defaultParams.featureNoToFriendlyName = x
      } text ("path to featureNoToFriendlyName")
      opt[Boolean]("expandTreeEnsemble") optional() foreach { x =>
        defaultParams.expandTreeEnsemble = x
      } text (s"expandTreeEnsemble")
      opt[String]("featureIniFile") optional() foreach { x =>
        defaultParams.featureIniFile = x
      } text (s"path to featureIniFile")
      opt[String]("gainTableStr") required() foreach { x =>
        defaultParams.gainTableStr = x
      } text (s"gainTableStr parameters")
      opt[String]("algo") optional() foreach { x =>
        defaultParams.algo = x
      } text (s"algorithm (${Algo.values.mkString(",")}), default: ${defaultParams.algo}")
      opt[String]("maxDepth") optional() foreach { x =>
        defaultParams.maxDepth = x.split(":").map(_.toInt)
      } text (s"max depth of the tree, default: ${defaultParams.maxDepth}")

      opt[Int]("numLeaves") optional() foreach { x =>
        defaultParams.numLeaves = x
      } text (s"num of leaves per tree, default: ${defaultParams.numLeaves}. Take precedence over --maxDepth.")
      opt[String]("numPruningLeaves") optional() foreach { x =>
        defaultParams.numPruningLeaves = x.split(":").map(_.toInt)
      } text (s"num of leaves per tree after pruning, default: ${defaultParams.numPruningLeaves}.")
      opt[String]("numIterations") optional() foreach { x =>
        defaultParams.numIterations = x.split(":").map(_.toInt)
      } text (s"number of iterations of boosting," + s" default: ${defaultParams.numIterations}")
      opt[String]("minInstancesPerNode") optional() foreach { x =>
        defaultParams.minInstancesPerNode = x.split(":").map(_.toInt)
      } text (s"the minimum number of documents allowed in a leaf of the tree, default: ${defaultParams.minInstancesPerNode}")
      opt[Int]("maxSplits") optional() foreach { x =>
        defaultParams.maxSplits = x
      } text (s"max Nodes to be split simultaneously, default: ${defaultParams.maxSplits}") validate { x =>
        if (x > 0 && x <= 512) success else failure("value <maxSplits> incorrect; should be between 1 and 512.")
      }
      opt[String]("learningRate") optional() foreach { x =>
        defaultParams.learningRate = x.split(":").map(_.toDouble)
      } text (s"learning rate of the score update, default: ${defaultParams.learningRate}")
      opt[Int]("testSpan") optional() foreach { x =>
        defaultParams.testSpan = x
      } text (s"test span")
      opt[Int]("numPartitions") optional() foreach { x =>
        defaultParams.numPartitions = x
      } text (s"number of partitions, default: ${defaultParams.numPartitions}")
      opt[Double]("sampleFeaturePercent") optional() foreach { x =>
        defaultParams.sampleFeaturePercent = x
      } text (s"global feature percentage used for training")
      opt[Double]("sampleQueryPercent") optional() foreach { x =>
        defaultParams.sampleQueryPercent = x
      } text (s"global query percentage used for training")
      opt[Double]("sampleDocPercent") optional() foreach { x =>
        defaultParams.sampleDocPercent = x
      } text (s"global doc percentage used for classification")
      opt[Double]("ffraction") optional() foreach { x =>
        defaultParams.ffraction = x
      } text (s"feature percentage used for training for each tree")
      opt[Double]("sfraction") optional() foreach { x =>
        defaultParams.sfraction = x
      } text (s"sample percentage used for training for each tree")
      opt[Double]("secondaryMS") optional() foreach { x =>
        defaultParams.secondaryMS = x
      } text (s"secondaryMetricShare")
      opt[Boolean]("secondaryLE") optional() foreach { x =>
        defaultParams.secondaryLE = x
      } text (s"secondaryIsoLabelExclusive")
      opt[Double]("sigma") optional() foreach { x =>
        defaultParams.sigma = x
      } text (s"parameter for init sigmoid table")
      opt[Boolean]("dw") optional() foreach { x =>
        defaultParams.distanceWeight2 = x
      } text (s"Distance weight 2 adjustment to cost")
      opt[String]("bafn") optional() foreach { x =>
        defaultParams.baselineAlphaFilename = x
      } text (s"Baseline alpha for tradeoffs of risk (0 is normal training)")
      opt[Double]("entropyCoefft") optional() foreach { x =>
        defaultParams.entropyCoefft = x
      } text (s"The entropy (regularization) coefficient between 0 and 1")
      opt[Double]("ffup") optional() foreach { x =>
        defaultParams.featureFirstUsePenalty = x
      } text (s"The feature first use penalty coefficient")
      opt[Double]("frup") optional() foreach { x =>
        defaultParams.featureReusePenalty = x
      } text (s"The feature re-use penalty (regularization) coefficient")
      opt[String]("learningStrategy") optional() foreach { x =>
        defaultParams.learningStrategy = x
      } text (s"learningStrategy for adaptive gradient descent")
      opt[String]("oNDCG") optional() foreach { x =>
        defaultParams.outputNdcgFilename = x
      } text (s"save ndcg of training phase in this file")
      opt[Boolean]("allr") optional() foreach { x =>
        defaultParams.active_lambda_learningStrategy = x
      } text (s"active lambda learning strategy or not")
      opt[Double]("rhol") optional() foreach { x =>
        defaultParams.rho_lambda = x
      } text (s"rho lambda value")
      opt[Boolean]("alvst") optional() foreach { x =>
        defaultParams.active_leaves_value_learningStrategy = x
      } text (s"active leave value learning strategy or not")
      opt[Double]("rholv") optional() foreach { x =>
        defaultParams.rho_leave = x
      } text (s"rho parameter for leave value learning strategy")
      opt[Boolean]("gn") optional() foreach { x =>
        defaultParams.GainNormalization = x
      } text (s"normalize the gian value in the comment or not")
      opt[String]("f2nf") optional() foreach { x =>
        defaultParams.feature2NameFile = x
      } text (s"path to feature to name map file")
      opt[Int]("vs") optional() foreach { x =>
        defaultParams.validationSpan = x
      } text (s"validation span")
      opt[Boolean]("es") optional() foreach { x =>
        defaultParams.useEarlystop = x
      } text (s"apply early stop or not")
      opt[String]("sgfn") optional() foreach { x =>
        defaultParams.secondGainsFileName = x
      }
      opt[String]("simdfn") optional() foreach { x =>
        defaultParams.secondaryInverseMaxDcgFileName = x
      }
      opt[String]("swfn") optional() foreach { x =>
        defaultParams.sampleWeightsFilename = x
      }
      opt[String]("bdfn") optional() foreach { x =>
        defaultParams.baselineDcgsFilename = x
      }
      opt[String]("dfn") optional() foreach { x =>
        defaultParams.discountsFilename = x
      }
    }
    parser.parse(args)
    run(defaultParams)
  }

  def run(params: Params) {
    require(params.numIterations.length == params.learningRate.length &&
      params.numIterations.length == params.minInstancesPerNode.length &&
      params.numIterations.length == params.numPruningLeaves.length,
      s"numiterations: ${params.numIterations}, learningRate: ${params.learningRate}, " +
        s"and minInstancesPerNode: ${params.minInstancesPerNode}, numPruningLeaves: ${params.numPruningLeaves}do not match")

    require(!params.maxDepth.exists(dep => dep > 30), s"value maxDepth:${params.maxDepth} incorrect; should be less than or equals to 30.")


    println(s"LambdaMARTRunner with parameters:\n${params.toString}")
    val conf = new SparkConf().setAppName(s"LambdaMARTRunner with $params")
    if (params.numPartitions != 0)
      conf.set("lambdaMart_numPartitions", s"${params.numPartitions}")
    val sc = new SparkContext(conf)
    try {
      //load training data
      var label = dataSetLoader.loadlabelScores(sc, params.label)
      val numSamples = label.length
      println(s"numSamples: $numSamples")

      var initScores = if (params.initScores == null) {
        new Array[Double](numSamples)
      } else {
        val loaded = dataSetLoader.loadInitScores(sc, params.initScores)
        require(loaded.length == numSamples, s"lengthOfInitScores: ${loaded.length} != numSamples: $numSamples")
        loaded
      }
      var queryBoundy = if (params.queryBoundy != null) dataSetLoader.loadQueryBoundy(sc, params.queryBoundy) else null
      require(queryBoundy == null || queryBoundy.last == numSamples, s"QueryBoundy ${queryBoundy.last} does not match with data $numSamples !")
      val numQuery = if (queryBoundy != null) queryBoundy.length - 1 else 0
      println(s"num of data query: $numQuery")

      val numSampleQuery = if (params.sampleQueryPercent < 1) (numQuery * params.sampleQueryPercent).toInt else numQuery
      println(s"num of sampling query: $numSampleQuery")
      val sampleQueryId = if (params.sampleQueryPercent < 1) {
        (new Random(Random.nextInt)).shuffle((0 until queryBoundy.length - 1).toList).take(numSampleQuery).toArray
      } else null //query index for training

      if (params.algo=="LambdaMart"&&params.sampleQueryPercent < 1) {
        // sampling
        label = dataSetLoader.getSampleLabels(sampleQueryId, queryBoundy, label)
        println(s"num of sampling labels: ${label.length}")
        initScores = dataSetLoader.getSampleInitScores(sampleQueryId, queryBoundy, initScores, label.length)
        require(label.length == initScores.length, s"num of labels ${label.length} does not match with initScores ${initScores.length}!")
      }
      else if(params.algo=="Classification" && params.sampleDocPercent!=1){
        val numDocSampling = (params.sampleDocPercent * numSamples).toInt
        if (params.sampleDocPercent<1) {
          label = label.take(numDocSampling)
          initScores = initScores.take(numDocSampling)
        }
        else{
          val newLabel = new Array[Short](numDocSampling)
          val newScores = new Array[Double](numDocSampling)
          var is = 0
          while(is<numDocSampling){
            newLabel(is)=label(is%numSamples)
            newScores(is)=label(is%numSamples)
            is+=1
          }
          label = newLabel
          initScores = newScores
        }
        println(s"num of sampling labels: ${label.length}")
        require(label.length == initScores.length, s"num of labels ${label.length} does not match with initScores ${initScores.length}!")
      }

      var trainingData =
      if(params.algo=="LambdaMart")
          dataSetLoader.loadTrainingDataForLambdamart(sc, params.trainingData, sc.defaultMinPartitions, sampleQueryId, queryBoundy, label.length)
      else
        dataSetLoader.loadTrainingDataForClassification(sc, params.trainingData, sc.defaultMinPartitions, label.length)

      if (params.sampleQueryPercent < 1) {
        queryBoundy = dataSetLoader.getSampleQueryBound(sampleQueryId, queryBoundy)
        require(queryBoundy.last == label.length, s"QueryBoundy ${queryBoundy.last} does not match with data ${label.length} !")
      }
      val numFeats = trainingData.count().toInt
      println(s"numFeats: $numFeats")
//      val numNonZeros = trainingData.map { x => x._2.default }.filter(_ != 0).count()
//      println(s"numFeats sparse on nonZero: $numNonZeros")

      val trainingData_T = genTransposedData(trainingData, numFeats, label.length)

      trainingData = dataSetLoader.getSampleFeatureData(sc, trainingData, params.sampleFeaturePercent)

      if (params.algo == "Classification") {
        label = label.map(x => (x * 2 - 1).toShort)
      }

      val trainingDataSet = new dataSet(label, initScores, queryBoundy, trainingData)

      var validtionDataSet: dataSet = null
      if (params.validationData != null) {
        val validationData = dataSetLoader.loadDataTransposed(sc, params.validationData)
        val labelV = dataSetLoader.loadlabelScores(sc, params.labelValidate)
        val initScoreV = if (params.initScoreValidate == null) {
          new Array[Double](labelV.length)
        }
        else {
          dataSetLoader.loadInitScores(sc, params.initScoreValidate)
        }

        val queryBoundyV = if (params.queryBoundy != null) dataSetLoader.loadQueryBoundy(sc, params.queryBoundyValidate) else null

        validtionDataSet = new dataSet(labelV, initScoreV, queryBoundyV, dataTransposed = validationData)
      }
      println(s"validationDataSet: $validtionDataSet")


      val gainTable = params.gainTableStr.split(':').map(_.toDouble)

      val boostingStrategy = config.BoostingStrategy.defaultParams(params.algo)
      boostingStrategy.treeStrategy.maxDepth = params.maxDepth(0)


      //extract secondGain and secondInverseMaxDcg
      if (params.secondGainsFileName != null && params.secondaryInverseMaxDcgFileName != null) {
        val spTf_1 = sc.textFile(params.secondGainsFileName)
        if (spTf_1.count() > 0)
          params.secondGains = spTf_1.first().split(",").map(_.toDouble)
        val spTf_2 = sc.textFile(params.secondaryInverseMaxDcgFileName)
        if (spTf_2.count() > 0)
          params.secondaryInverseMaxDcg = spTf_2.first().split(",").map(_.toDouble)
      }

      if (params.discountsFilename != null) {
        val spTf = sc.textFile(params.discountsFilename)
        if (spTf.count() > 0)
          params.discounts = spTf.first().split(",").map(_.toDouble)
      }

      if (params.sampleWeightsFilename != null) {
        val spTf = sc.textFile(params.sampleWeightsFilename)
        if (spTf.count() > 0)
          params.sampleWeights = spTf.first().split(",").map(_.toDouble)
      }

      if (params.baselineDcgsFilename != null) {
        val spTf = sc.textFile(params.baselineDcgsFilename)
        if (spTf.count() > 0)
          params.baselineDcgs = spTf.first().split(",").map(_.toDouble)
      }

      if (params.baselineAlphaFilename != null) {
        val spTf = sc.textFile(params.baselineAlphaFilename)
        if (spTf.count() > 0)
          params.baselineAlpha = spTf.first().split(",").map(_.toDouble)
      }

      val feature2Gain = new Array[Double](numFeats)
      if (params.algo == "LambdaMart" || params.algo == "Classification") {
        val startTime = System.nanoTime()
        val model = LambdaMART.train(trainingDataSet, validtionDataSet, trainingData_T, gainTable,
          boostingStrategy, params, feature2Gain)
        val elapsedTime = (System.nanoTime() - startTime) / 1e9
        println(s"Training time: $elapsedTime seconds")

        // test
        if (params.algo == "LambdaMart" && params.testSpan != 0) {
          val testNDCG = testModel(sc, model, params, gainTable)
          println(s"testNDCG error 0 = " + testNDCG(0))
          for (i <- 1 until testNDCG.length) {
            val it = i * params.testSpan
            println(s"testNDCG error $it = " + testNDCG(i))
          }
        }
        else if (params.algo == "Classification" && params.testData != null) {
          val testData = MLUtils.loadLibSVMFile(sc, params.testData)
          println(s"numSamples: ${testData.count()}")
          val scoreAndLabels = testData.map { point =>
            val predictions =  model.trees.map(_.predict(point.features))
            val claPred = new Array[Double](predictions.length)
            Range(1,predictions.length).foreach{it=>
              predictions(it)+=predictions(it-1)
              if(predictions(it)>=0) claPred(it)=1.0
              else claPred(it)=0.0
            }

            (claPred, point.label)
          }

          Range(0,model.trees.length, params.testSpan).foreach{it=>
            val scoreLabel = scoreAndLabels.map{case(claPred, lb)=>(claPred(it),lb)}
            val metrics = new BinaryClassificationMetrics(scoreLabel)
            val accuracy = metrics.areaUnderROC()
            println(s"Accuracy $it = $accuracy")
          }
        }

        //        if (params.featureIniFile != null) {
        //          val featureIniPath = new Path(params.featureIniFile)
        //          val featurefs = TreeUtils.getFileSystem(trainingData.context.getConf, featureIniPath)
        //          featurefs.copyToLocalFile(false, featureIniPath, new Path("treeEnsemble.ini"))
        //        }

        for (i <- 0 until model.trees.length) {
          val evaluator = model.trees(i)
          evaluator.sequence("treeEnsemble.ini", evaluator, i + 1)
        }
        println(s"save succeed")
        val totalEvaluators = model.trees.length
        val evalNodes = Array.tabulate[Int](totalEvaluators)(_ + 1)
        treeAggregatorFormat.appendTreeAggregator(params.expandTreeEnsemble, "treeEnsemble.ini", totalEvaluators + 1, evalNodes)

        if (params.feature2NameFile != null) {
          val feature2Name = dataSetLoader.loadFeature2NameMap(sc, params.feature2NameFile)
          treeAggregatorFormat.toCommentFormat("treeEnsemble.ini", params, feature2Name, feature2Gain)
        }
        val outPath = new Path(params.outputTreeEnsemble)
        val fs = TreeUtils.getFileSystem(trainingData.context.getConf, outPath)
        fs.copyFromLocalFile(false, true, new Path("treeEnsemble.ini"), outPath)

        if (model.totalNumNodes < 30) {
          println(model.toDebugString) // Print full model.
        } else {
          println(model) // Print model summary.
        }
        // val testMSE = meanSquaredError(model, testData)
        // println(s"Test mean squared error = $testMSE")
      }
    } finally {
      sc.stop()
    }
  }



  def loadTestData(sc: SparkContext, path: String): RDD[Vector] = {
    sc.textFile(path).map { line =>
      if (line.contains("#"))
        Vectors.dense(line.split("#")(1).split(",").map(_.toDouble))
      else
        Vectors.dense(line.split(",").map(_.toDouble))
    }


    //    val testData = sc.textFile(path).map {line => line.split("#")(1).split(",").map(_.toDouble)}
    //    val testTrans = testData.zipWithIndex.flatMap {
    //      case (row, rowIndex) => row.zipWithIndex.map {
    //        case (number, columnIndex) => columnIndex -> (rowIndex, number)
    //      }
    //    }
    //    val testT = testTrans.groupByKey.sortByKey().values
    //      .map {
    //        indexedRow => indexedRow.toSeq.sortBy(_._1).map(_._2).toArray
    //      }
    //    testT.map(line => Vectors.dense(line))
  }

  def genTransposedData(trainingData: RDD[(Int, SparseArray[Short], Array[SplitInfo])],
                        numFeats: Int,
                        numSamples: Int): RDD[(Int, Array[Array[Short]])] = {
    println("generating transposed data...")
    // validate that the original data is ordered
    val denseAsc = trainingData.mapPartitions { iter =>
      var res = Iterator.single(true)
      if (iter.hasNext) {
        var prev = iter.next()._1
        val remaining = iter.dropWhile { case (fi, _, _) =>
          val goodNext = fi - prev == 1
          prev = fi
          goodNext
        }
        res = Iterator.single(!remaining.hasNext)
      }
      res
    }.reduce(_ && _)
    assert(denseAsc, "the original data must be ordered.")
    println("pass data check in transposing")

    val numPartitions = trainingData.partitions.length
    val (siMinPP, lcNumSamplesPP) = TreeUtils.getPartitionOffsets(numSamples, numPartitions)
    val trainingData_T = trainingData.mapPartitions { iter =>
      val (metaIter, dataIter) = iter.duplicate
      val fiMin = metaIter.next()._1
      val lcNumFeats = metaIter.length + 1
      val blocksPP = Array.tabulate(numPartitions)(pi => Array.ofDim[Short](lcNumFeats, lcNumSamplesPP(pi)))
      dataIter.foreach { case (fi, sparseSamples, _) =>
        val samples = sparseSamples.toArray
        val lfi = fi - fiMin
        var pi = 0
        while (pi < numPartitions) {
          Array.copy(samples, siMinPP(pi), blocksPP(pi)(lfi), 0, lcNumSamplesPP(pi))
          pi += 1
        }
      }
      Range(0, numPartitions).iterator.map(pi => (pi, (fiMin, blocksPP(pi))))
    }.partitionBy(new HashPartitioner(numPartitions)).mapPartitionsWithIndex((pid, iter) => {
      val siMin = siMinPP(pid)
      val sampleSlice = new Array[Array[Short]](numFeats)
      iter.foreach { case (_, (fiMin, blocks)) =>
        var lfi = 0
        while (lfi < blocks.length) {
          sampleSlice(lfi + fiMin) = blocks(lfi)
          lfi += 1
        }
      }
      Iterator.single((siMin, sampleSlice))
    }, preservesPartitioning = true)
    trainingData_T.persist(StorageLevel.MEMORY_AND_DISK).setName("trainingData_T").count()
    trainingData_T
  }

  def testModel(sc: SparkContext, model: GradientBoostedDecisionTreesModel, params: Params, gainTable: Array[Double]): Array[Double] = {
    val testData = loadTestData(sc, params.testData).cache().setName("TestData")
    println(s"numTestFeature: ${testData.first().toArray.length}")
    val numTest = testData.count()
    println(s"numTest: $numTest")
    val testLabels = dataSetLoader.loadlabelScores(sc, params.testLabel)

    println(s"numTestLabels: ${testLabels.length}")
    require(testLabels.length == numTest, s"lengthOfLabels: ${testLabels.length} != numTestSamples: $numTest")
    val testQueryBound = dataSetLoader.loadQueryBoundy(sc, params.testQueryBound)
    require(testQueryBound.last == numTest, s"TestQueryBoundy ${testQueryBound.last} does not match with test data $numTest!")

    val rate = params.testSpan
    val predictions = testData.map { features =>
      val scores = model.trees.map(_.predict(features))
      for (it <- 1 until model.trees.length) {
        scores(it) += scores(it - 1)
      }

      scores.zipWithIndex.collect {
        case (score, it) if it == 0 || (it + 1) % rate == 0 => score
      }
    }
    val predictionsByIter = predictions.zipWithIndex.flatMap {
      case (row, rowIndex) => row.zipWithIndex.map {
        case (number, columnIndex) => columnIndex ->(rowIndex, number)
      }
    }.groupByKey.sortByKey().values
      .map {
        indexedRow => indexedRow.toArray.sortBy(_._1).map(_._2)
      }

    val learningRates = params.learningRate
    val distanceWeight2 = params.distanceWeight2
    val baselineAlpha = params.baselineAlpha
    val secondMs = params.secondaryMS
    val secondLe = params.secondaryLE
    val secondGains = params.secondGains
    val secondaryInverseMacDcg = params.secondaryInverseMaxDcg
    val discounts = params.discounts
    val baselineDcgs = params.baselineDcgs
    val dc = new DerivativeCalculator
    dc.init(testLabels, gainTable, testQueryBound,
      learningRates(0), distanceWeight2, baselineAlpha,
      secondMs, secondLe, secondGains, secondaryInverseMacDcg, discounts, baselineDcgs)
    val numQueries = testQueryBound.length - 1
    val dcBc = sc.broadcast(dc)

    predictionsByIter.map { scores =>
      val dc = dcBc.value
      dc.getPartErrors(scores, 0, numQueries) / numQueries
    }.collect()

  }

  /** *
    * def meanSquaredError(
    * model: { def predict(features: Vector): Double },
    * data: RDD[LabeledPoint]): Double = {
    * data.map { y =>
    * val err = model.predict(y.features) - y.label
    * err * err
    * }.mean()
    * } ***/
}
