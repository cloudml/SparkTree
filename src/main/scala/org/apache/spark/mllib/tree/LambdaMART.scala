package org.apache.spark.mllib.tree

//import akka.io.Udp.SO.Broadcast
import java.io.{File, FileOutputStream, PrintWriter}

import breeze.linalg.min
import org.apache.hadoop.fs.Path
import org.apache.spark.Logging
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.examples.mllib.LambdaMARTRunner.Params
import org.apache.spark.mllib.dataSet.dataSet
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.tree.config.Algo._
import org.apache.spark.mllib.tree.config.BoostingStrategy
import org.apache.spark.mllib.tree.impl.TimeTracker
import org.apache.spark.mllib.tree.impurity.Variance
import org.apache.spark.mllib.tree.model.ensemblemodels.GradientBoostedDecisionTreesModel
import org.apache.spark.mllib.tree.model.opdtmodel.OptimizedDecisionTreeModel
import org.apache.spark.mllib.util.TreeUtils
import org.apache.spark.rdd.RDD

import scala.util.Random

class LambdaMART(val boostingStrategy: BoostingStrategy,
                 val params: Params) extends Serializable with Logging {
  def run(trainingDataSet: dataSet,
          validateDataSet: dataSet,
          trainingData_T: RDD[(Int, Array[Array[Short]])],
          gainTable: Array[Double],
          feature2Gain: Array[Double]): GradientBoostedDecisionTreesModel = {
    val algo = boostingStrategy.treeStrategy.algo

    algo match {
      case LambdaMart =>
        LambdaMART.boostMart(trainingDataSet, validateDataSet, trainingData_T, gainTable,
          boostingStrategy, params, feature2Gain)
      case Regression =>
        LambdaMART.boostRegression(trainingDataSet, validateDataSet, trainingData_T, gainTable,
          boostingStrategy, params, feature2Gain)
      case Classification =>

        LambdaMART.boostRegression(trainingDataSet, validateDataSet, trainingData_T, gainTable,
          boostingStrategy, params, feature2Gain)
      case _ =>
        throw new IllegalArgumentException(s"$algo is not supported by the implementation of LambdaMART.")
    }
  }
}

object LambdaMART extends Logging {
  def train(trainingDataSet: dataSet,
            validateDataSet: dataSet,
            trainingData_T: RDD[(Int, Array[Array[Short]])],
            gainTable: Array[Double],
            boostingStrategy: BoostingStrategy,
            params: Params,
            feature2Gain: Array[Double]): GradientBoostedDecisionTreesModel = {

    new LambdaMART(boostingStrategy, params)
      .run(trainingDataSet, validateDataSet, trainingData_T, gainTable, feature2Gain)
  }

  private def boostMart(trainingDataSet: dataSet,
                        validateDataSet: dataSet,
                        trainingData_T: RDD[(Int, Array[Array[Short]])],
                        gainTable: Array[Double],
                        boostingStrategy: BoostingStrategy,
                        params: Params,
                        feature2Gain: Array[Double]): GradientBoostedDecisionTreesModel = {
    val timer = new TimeTracker()
    timer.start("total")
    timer.start("init")

    boostingStrategy.assertValid()
    val learningStrategy = params.learningStrategy
    // Initialize gradient boosting parameters
    val numPhases = params.numIterations.length
    val numTrees = params.numIterations.sum //different phases different trees number.
    var baseLearners = new Array[OptimizedDecisionTreeModel](numTrees)
    var baseLearnerWeights = new Array[Double](numTrees)
    // val loss = boostingStrategy.loss
    val numPruningLeaves = params.numPruningLeaves

    // Prepare strategy for individual trees, which use regression with variance impurity.
    val treeStrategy = boostingStrategy.treeStrategy.copy
    // val validationTol = boostingStrategy.validationTol
    treeStrategy.algo = LambdaMart
    treeStrategy.impurity = Variance
    treeStrategy.assertValid()


    val trainingData = trainingDataSet.getData()
    val label = trainingDataSet.getLabel()
    val queryBoundy = trainingDataSet.getQueryBoundy()
    val initScores = trainingDataSet.getScore()

    val sc = trainingData.sparkContext
    val numSamples = label.length
    val numQueries = queryBoundy.length - 1
    val (qiMinPP, lcNumQueriesPP) = TreeUtils.getPartitionOffsets(numQueries, sc.defaultParallelism)
    //println(">>>>>>>>>>>")
    //println(qiMinPP.mkString(","))
    //println(lcNumQueriesPP.mkString(","))
    val pdcRDD = sc.parallelize(qiMinPP.zip(lcNumQueriesPP)).cache().setName("PDCCtrl")

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
    //sigma = params.learningRate(0)
    dc.init(label, gainTable, queryBoundy,
      learningRates(0), distanceWeight2, baselineAlpha,
      secondMs, secondLe, secondGains, secondaryInverseMacDcg, discounts, baselineDcgs)

    val dcBc = sc.broadcast(dc)
    val lambdas = new Array[Double](numSamples)
    val weights = new Array[Double](numSamples)

    timer.stop("init")

    val currentScores = initScores
    val initErrors = evaluateErrors(pdcRDD, dcBc, currentScores, numQueries)
    println(s"NDCG initError sum = $initErrors")

    var m = 0
    var numIterations = 0

    var earlystop = false
    val useEarlystop = params.useEarlystop
    var phase = 0
    val oldRep = new Array[Double](numSamples)
    val validationSpan = params.validationSpan
    val multiplier_Score = 1.0
    var sampleFraction = params.sfraction

    while (phase < numPhases && !earlystop) {
      numIterations += params.numIterations(phase)
      //initial derivativeCalculator for every phase
      val dcPhase = new DerivativeCalculator
      dcPhase.init(label, gainTable, queryBoundy, learningRates(phase),
        distanceWeight2, baselineAlpha,
        secondMs, secondLe, secondGains, secondaryInverseMacDcg, discounts, baselineDcgs)

      val dcPhaseBc = sc.broadcast(dcPhase)

      var qfraction = 0.5 // stochastic sampling fraction of query per tree
      while (m < numIterations && !earlystop) {
        timer.start(s"building tree $m")
        println("\nGradient boosting tree iteration " + m)


        //        println(s"active samples: ${activeSamples.sum}")

        val iterationBc = sc.broadcast(m)
        val currentScoresBc = sc.broadcast(currentScores)
        updateDerivatives(pdcRDD, dcPhaseBc, currentScoresBc, iterationBc, lambdas, weights)
        currentScoresBc.unpersist(blocking = false)
        iterationBc.unpersist(blocking = false)

        //adaptive lambda
        if (params.active_lambda_learningStrategy) {
          val rho_lambda = params.rho_lambda
          if (learningStrategy == "sgd") {

          }
          else if (learningStrategy == "momentum") {
            Range(0, numSamples).par.foreach { si =>
              lambdas(si) = rho_lambda * oldRep(si) + lambdas(si)
              oldRep(si) = lambdas(si)
            }
          }
          //          else if (learningStrategy == "adagrad") {
          //            Range(0, numSamples).par.foreach { si =>
          //              oldRep(si) += lambdas(si) * lambdas(si)
          //              lambdas(si) = lambdas(si) / math.sqrt(oldRep(si) + 1e-9)
          //            }
          //          }
          //          else if (learningStrategy == "adadelta") {
          //            Range(0, numSamples).par.foreach { si =>
          //              oldRep(si) = rho_lambda * oldRep(si) + (1 - rho_lambda) * lambdas(si) * lambdas(si)
          //              lambdas(si) = learningRate(phase) * lambdas(si) / scala.math.sqrt(oldRep(si) + 1e-9)
          //            }
          //          }
        }

        val lambdasBc = sc.broadcast(lambdas)
        val weightsBc = sc.broadcast(weights)


        sampleFraction = min(params.sfraction + (1 - params.sfraction) / 10 * (m * 11.0 / numIterations).toInt, 1.0)
        println(s"sampleFraction: $sampleFraction")
        qfraction = sampleFraction
        val initTimer = new TimeTracker()
        initTimer.start("topInfo")
        val (activeSamples, sumCount, sumTarget, sumSquare, sumWeight): (Array[Byte], Int, Double, Double, Double) = {
          if (qfraction >= 1.0) {
            (Array.fill[Byte](numSamples)(1), numSamples, lambdas.sum, lambdas.map(x => x * x).sum, weights.sum)
          }
          else {
            val act = new Array[Byte](numSamples)
            val (sumC, sumT, sumS, sumW): (Int, Double, Double, Double) = updateActSamples(pdcRDD, dcPhaseBc, lambdasBc, weightsBc, qfraction, act)
            (act, sumC, sumT, sumS, sumW)
          }
        }
        println(s"sampleCount: $sumCount")
        initTimer.stop("topInfo")
        println(s"$initTimer")


        logDebug(s"Iteration $m: scores: \n" + currentScores.mkString(" "))

        val featureUseCount = new Array[Int](feature2Gain.length)
        var TrainingDataUse = trainingData
        if (params.ffraction < 1.0) {
          TrainingDataUse = trainingData.filter(item => IsSeleted(params.ffraction))
        }

        treeStrategy.maxDepth = params.maxDepth(phase)
        val tree = new LambdaMARTDecisionTree(treeStrategy, params.minInstancesPerNode(phase),
          params.numLeaves, params.maxSplits, params.expandTreeEnsemble)
        val (model, treeScores) = tree.run(TrainingDataUse, trainingData_T, lambdasBc, weightsBc, numSamples,
          params.entropyCoefft, featureUseCount, params.featureFirstUsePenalty,
          params.featureReusePenalty, feature2Gain, params.sampleWeights, numPruningLeaves(phase),
          (sumCount, sumTarget, sumSquare, sumWeight), actSamples = activeSamples)
        lambdasBc.unpersist(blocking = false)
        weightsBc.unpersist(blocking = false)
        timer.stop(s"building tree $m")

        baseLearners(m) = model
        baseLearnerWeights(m) = learningRates(phase)

        Range(0, numSamples).par.foreach(si =>
          currentScores(si) += baseLearnerWeights(m) * treeScores(si)
        )
        //testing continue training


        /**
          * //adaptive leaves value
          * if(params.active_leaves_value_learningStrategy){
          * val rho_leave = params.rho_leave
          * if(learningStrategy == "sgd") {
          * Range(0, numSamples).par.foreach(si =>
          * currentScores(si) += learningRate(phase) * treeScores(si)
          * )
          * }
          * else if(learningStrategy == "momentum"){
          * Range(0, numSamples).par.foreach { si =>
          * val deltaScore = rho_leave * oldRep(si) + learningRate(phase) * treeScores(si)
          * currentScores(si) += deltaScore
          * oldRep(si) = deltaScore
          * }
          * }
          * else if (learningStrategy == "adagrad"){
          * Range(0, numSamples).par.foreach { si =>
          * oldRep(si) += treeScores(si) * treeScores(si)
          * currentScores(si) += learningRate(phase) * treeScores(si) / math.sqrt(oldRep(si) + 1e-9)
          * }
          * }
          * else if (learningStrategy == "adadelta"){
          * Range(0, numSamples).par.foreach { si =>
          * oldRep(si) = rho_leave * oldRep(si) + (1- rho_leave)*treeScores(si)*treeScores(si)
          * currentScores(si) += learningRate(phase) * treeScores(si) / math.sqrt(oldRep(si) + 1e-9)
          * }
          * }
          * } ***/


        //validate the model
        // println(s"validationDataSet: $validateDataSet")

        if (validateDataSet != null && 0 == (m % validationSpan) && useEarlystop) {
          val numQueries_V = validateDataSet.getQueryBoundy().length - 1
          val (qiMinPP_V, lcNumQueriesPP_V) = TreeUtils.getPartitionOffsets(numQueries_V, sc.defaultParallelism)
          //println(s"")
          val pdcRDD_V = sc.parallelize(qiMinPP_V.zip(lcNumQueriesPP_V)).cache().setName("PDCCtrl_V")

          val dc_v = new DerivativeCalculator
          dc_v.init(validateDataSet.getLabel(), gainTable, validateDataSet.getQueryBoundy(),
            learningRates(phase), params.distanceWeight2, baselineAlpha,
            secondMs, secondLe, secondGains, secondaryInverseMacDcg, discounts, baselineDcgs)

          val currentBaseLearners = new Array[OptimizedDecisionTreeModel](m + 1)
          val currentBaselearnerWeights = new Array[Double](m + 1)
          baseLearners.copyToArray(currentBaseLearners, 0, m + 1)
          baseLearnerWeights.copyToArray(currentBaselearnerWeights, 0, m + 1)
          val currentModel = new GradientBoostedDecisionTreesModel(Regression, currentBaseLearners, currentBaselearnerWeights)
          val currentValidateScore = new Array[Double](validateDataSet.getLabel().length)

          //val currentValidateScore_Bc = sc.broadcast(currentValidateScore)
          val currentModel_Bc = sc.broadcast(currentModel)

          validateDataSet.getDataTransposed().map { item =>
            (item._1, currentModel_Bc.value.predict(item._2))
          }.collect().foreach { case (sid, score) =>
            currentValidateScore(sid) = score
          }

          println(s"currentScores: ${currentValidateScore.mkString(",")}")

          val errors = evaluateErrors(pdcRDD_V, sc.broadcast(dc_v), currentValidateScore, numQueries_V)

          println(s"validation errors: $errors")

          if (errors < 1.0e-6) {
            earlystop = true
            baseLearners = currentBaseLearners
            baseLearnerWeights = currentBaselearnerWeights
          }
          currentModel_Bc.unpersist(blocking = false)
        }
        val errors = evaluateErrors(pdcRDD, dcPhaseBc, currentScores, numQueries)

        val pw = new PrintWriter(new FileOutputStream(new File("ndcg.txt"), true))
        pw.write(errors.toString + "\n")
        pw.close()

        println(s"NDCG error sum = $errors")
        println(s"length:" + model.topNode.internalNodes)
        // println("error of gbt = " + currentScores.iterator.map(re => re * re).sum / numSamples)
        //model.sequence("treeEnsemble.ini", model, m + 1)
        m += 1
      }
      phase += 1
    }

    timer.stop("total")

    if (params.outputNdcgFilename != null) {
      val outPath = new Path(params.outputNdcgFilename)
      val fs = TreeUtils.getFileSystem(trainingData.context.getConf, outPath)
      fs.copyFromLocalFile(true, true, new Path("ndcg.txt"), outPath)
    }
    println("Internal timing for LambdaMARTDecisionTree:")
    println(s"$timer")

    trainingData.unpersist(blocking = false)
    trainingData_T.unpersist(blocking = false)

    new GradientBoostedDecisionTreesModel(Regression, baseLearners, baseLearnerWeights)
  }

  private def boostRegression(trainingDataSet: dataSet,
                              validateDataSet: dataSet,
                              trainingData_T: RDD[(Int, Array[Array[Short]])],
                              gainTable: Array[Double],
                              boostingStrategy: BoostingStrategy,
                              params: Params,
                              feature2Gain: Array[Double]): GradientBoostedDecisionTreesModel = {
    val timer = new TimeTracker()
    timer.start("total")
    timer.start("init")

    boostingStrategy.assertValid()

    // Initialize gradient boosting parameters
    val numPhases = params.numIterations.length
    val numTrees = params.numIterations.sum //different phases different trees number.
    var baseLearners = new Array[OptimizedDecisionTreeModel](numTrees)
    var baseLearnerWeights = new Array[Double](numTrees)
    // val loss = boostingStrategy.loss


    // Prepare strategy for individual trees, which use regression with variance impurity.
    val treeStrategy = boostingStrategy.treeStrategy.copy
    // val validationTol = boostingStrategy.validationTol
    treeStrategy.algo = Regression
    treeStrategy.impurity = Variance
    treeStrategy.assertValid()
    val loss = boostingStrategy.loss

    val trainingData = trainingDataSet.getData()
    val label = trainingDataSet.getLabel()
    val initScores = trainingDataSet.getScore()

    val sc = trainingData.sparkContext
    val numSamples = label.length

    val learningRates = params.learningRate

    val lambdas = new Array[Double](numSamples)
    var ni = 0
    while (ni < numSamples) {
      lambdas(ni) = -2 * (label(ni) - initScores(ni))
      ni += 1
    }

    //    val weights = new Array[Double](numSamples)
    val weightsBc = sc.broadcast(Array.empty[Double])
    timer.stop("init")

    val currentScores = initScores
    var initErrors = 0.0
    ni = 0
    while (ni < numSamples) {
      initErrors += loss.computeError(currentScores(ni), label(ni))
      ni += 1
    }
    initErrors /= numSamples
    println(s"logloss initError sum = $initErrors")

    var m = 0
    var numIterations = 0

    var earlystop = false
    val useEarlystop = params.useEarlystop
    var phase = 0
    val oldRep = if (params.active_lambda_learningStrategy) new Array[Double](numSamples) else Array.empty[Double]
    val validationSpan = params.validationSpan
    //    val multiplier_Score = 1.0
    val learningStrategy = params.learningStrategy
    while (phase < numPhases && !earlystop) {
      numIterations += params.numIterations(phase)

      while (m < numIterations && !earlystop) {
        timer.start(s"building tree $m")
        println("\nGradient boosting tree iteration " + m)
        //update lambda
        Range(0, numSamples).par.foreach { ni =>
          lambdas(ni) = -loss.gradient(currentScores(ni), label(ni))
        }

        if (params.active_lambda_learningStrategy) {
          val rho_lambda = params.rho_lambda
          if (learningStrategy == "sgd") {
          }
          else if (learningStrategy == "momentum") {
            Range(0, numSamples).par.foreach { si =>
              lambdas(si) = rho_lambda * oldRep(si) + lambdas(si)
              oldRep(si) = lambdas(si)
            }
          }
          else if (learningStrategy == "adagrad") {
            Range(0, numSamples).par.foreach { si =>
              oldRep(si) += lambdas(si) * lambdas(si)
              lambdas(si) = lambdas(si) / math.sqrt(oldRep(si)+1.0)
            }
          }
          else if (learningStrategy == "adadelta") {
            Range(0, numSamples).par.foreach { si =>
              oldRep(si) = rho_lambda * oldRep(si) + (1 - rho_lambda) * lambdas(si) * lambdas(si)
              lambdas(si) = lambdas(si) / scala.math.sqrt(oldRep(si) + 1.0)

            }
          }
        }

        val lambdasBc = sc.broadcast(lambdas)

        logDebug(s"Iteration $m: scores: \n" + currentScores.mkString(" "))

        val featureUseCount = new Array[Int](trainingData.count().toInt)
        var TrainingDataUse = trainingData
        if (params.ffraction < 1.0) {
          TrainingDataUse = trainingData.filter(item => IsSeleted(params.ffraction))
        }

        var si = 0
        var sumLambda = 0.0
        var sumSquare = 0.0
        while (si < lambdas.length) {
          sumLambda += lambdas(si)
          sumSquare += lambdas(si) * lambdas(si)
          si += 1
        }
        val topValue = (numSamples, sumLambda, sumSquare, 0.0)
        val tree = new LambdaMARTDecisionTree(treeStrategy, params.minInstancesPerNode(phase),
          params.numLeaves, params.maxSplits, params.expandTreeEnsemble)
        val (model, treeScores) = tree.run(TrainingDataUse, trainingData_T, lambdasBc, weightsBc, numSamples,
          params.entropyCoefft, featureUseCount, params.featureFirstUsePenalty,
          params.featureReusePenalty, feature2Gain, params.sampleWeights, params.numPruningLeaves(phase), topValue)
        lambdasBc.unpersist(blocking = false)

        timer.stop(s"building tree $m")

        baseLearners(m) = model
        baseLearnerWeights(m) = learningRates(phase)
        println(s"learningRate: ${baseLearnerWeights(m)}")
        Range(0, numSamples).par.foreach(si =>
          currentScores(si) += baseLearnerWeights(m) * treeScores(si)
        )

        //validate the model
        // println(s"validationDataSet: $validateDataSet")

        if (validateDataSet != null && 0 == (m % validationSpan) && useEarlystop) {
          val currentBaseLearners = new Array[OptimizedDecisionTreeModel](m + 1)
          val currentBaselearnerWeights = new Array[Double](m + 1)
          baseLearners.copyToArray(currentBaseLearners, 0, m + 1)
          baseLearnerWeights.copyToArray(currentBaselearnerWeights, 0, m + 1)
          val currentModel = new GradientBoostedDecisionTreesModel(Regression, currentBaseLearners, currentBaselearnerWeights)
          val validateLabel = validateDataSet.getLabel()
          val numValidate = validateLabel.length

          //val currentValidateScore_Bc = sc.broadcast(currentValidateScore)
          val currentModel_Bc = sc.broadcast(currentModel)

          val currentValidateScore = validateDataSet.getDataTransposed().map { item =>
            currentModel_Bc.value.predict(item._2)
          }.collect()



          var errors = 0.0
          Range(0, numValidate).foreach { ni => val x = loss.computeError(currentValidateScore(ni), validateLabel(ni))
            errors += x
          }

          println(s"validation errors: $errors")

          if (errors < 1.0e-6) {
            earlystop = true
            baseLearners = currentBaseLearners
            baseLearnerWeights = currentBaselearnerWeights
          }
          currentModel_Bc.unpersist(blocking = false)
        }

        var errors = 0.0
        Range(0, numSamples).foreach { ni => val x = loss.computeError(currentScores(ni), label(ni))
          errors += x
        }
        errors /= numSamples

        val pw = new PrintWriter(new FileOutputStream(new File("ndcg.txt"), true))
        pw.write(errors.toString + "\n")
        pw.close()

        println(s"logloss error sum = $errors")
        println(s"length:" + model.topNode.internalNodes)
        // println("error of gbt = " + currentScores.iterator.map(re => re * re).sum / numSamples)
        //model.sequence("treeEnsemble.ini", model, m + 1)

        if (m % params.testSpan == 0) {
          val scoreAndLabels = new Array[(Double, Double)](10000)
          Range(0, 1000000, 100).foreach { it =>
            if (currentScores(it) >= 0)
              scoreAndLabels(it / 100) = (1.0, label(it).toDouble)
            else
              scoreAndLabels(it / 100) = (-1.0, label(it).toDouble)
          }
          val slRDD = sc.makeRDD(scoreAndLabels)
          val metrics = new BinaryClassificationMetrics(slRDD)
          val accuracy = metrics.areaUnderROC()
          println(s"test Accuracy at $m = $accuracy")
        }
        if (m % params.validationSpan == 0 || m == numIterations - 1) {
          val scoreAndLabels = new Array[(Double, Double)](label.length)
          var i = 0
          while (i < label.length) {
            if (currentScores(i) >= 0)
              scoreAndLabels(i) = (1.0, label(i).toDouble)
            else
              scoreAndLabels(i) = (-1.0, label(i).toDouble)
            i += 1
          }
          val slRDD = sc.makeRDD(scoreAndLabels)
          val metrics = new BinaryClassificationMetrics(slRDD)
          val accuracy = metrics.areaUnderROC()
          println(s"training Accuracy at $m = $accuracy")
        }

        m += 1
      }
      phase += 1
    }

    timer.stop("total")

    //    val scoreAndLabels = sc.makeRDD(currentScores.map(x=> if(x>0) 1.0 else -1.0).zip(label.map(_.toDouble)))
    //    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    //    val accuracy = metrics.areaUnderROC()
    //    println(s"Accuracy = $accuracy")

    println("Internal timing for RegressionDecisionTree:")
    println(s"$timer")
    weightsBc.unpersist(blocking = false)
    trainingData.unpersist(blocking = false)
    trainingData_T.unpersist(blocking = false)

    new GradientBoostedDecisionTreesModel(Regression, baseLearners, baseLearnerWeights)
  }


  def updateDerivatives(pdcRDD: RDD[(Int, Int)],
                        dcBc: Broadcast[DerivativeCalculator],
                        currentScoresBc: Broadcast[Array[Double]],
                        iterationBc: Broadcast[Int],
                        lambdas: Array[Double],
                        weights: Array[Double]): Unit = {
    val partDerivs = pdcRDD.mapPartitions { iter =>
      val dc = dcBc.value
      val currentScores = currentScoresBc.value
      val iteration = iterationBc.value
      iter.map { case (qiMin, lcNumQueries) =>
        dc.getPartDerivatives(currentScores, qiMin, qiMin + lcNumQueries, iteration)
      }
    }.collect()
    partDerivs.par.foreach { case (siMin, lcLambdas, lcWeights) =>
      Array.copy(lcLambdas, 0, lambdas, siMin, lcLambdas.length)
      Array.copy(lcWeights, 0, weights, siMin, lcWeights.length)
    }
  }

  def updateActSamples(pdcRDD: RDD[(Int, Int)], dcBc: Broadcast[DerivativeCalculator],
                       lambdasBc: Broadcast[Array[Double]],
                       weightsBc: Broadcast[Array[Double]],
                       fraction: Double,
                       act: Array[Byte]): (Int, Double, Double, Double) = {
    val partAct = pdcRDD.mapPartitions { iter =>
      val lambdas = lambdasBc.value
      val weights = weightsBc.value
      val queryBoundy = dcBc.value.queryBoundy
      val frac = fraction
      iter.map { case (qiMin, lcNumQueries) =>
        val qiEnd = qiMin + lcNumQueries
        val siTotalMin = queryBoundy(qiMin)
        val numTotalDocs = queryBoundy(qiEnd) - siTotalMin
        val lcActSamples = new Array[Byte](numTotalDocs)
        var lcSumCount = 0
        var lcSumTarget = 0.0
        var lcSumSquare = 0.0
        var lcSumWeight = 0.0
        var qi = qiMin
        while (qi < qiEnd) {
          val lcMin = queryBoundy(qi) - siTotalMin
          val siMin = queryBoundy(qi)
          val siEnd = queryBoundy(qi + 1)
          val numDocsPerQuery = siEnd - siMin
          if (Random.nextDouble() <= frac) {
            Range(lcMin, lcMin + numDocsPerQuery).foreach { lsi =>
              lcActSamples(lsi) = 1.toByte
              lcSumTarget += lambdas(siTotalMin + lsi)
              lcSumSquare += lambdas(siTotalMin + lsi) * lambdas(siTotalMin + lsi)
              lcSumWeight += weights(siTotalMin + lsi)
            }
            lcSumCount += numDocsPerQuery
          }
          else {
            Range(lcMin, lcMin + numDocsPerQuery).foreach { lsi =>
              lcActSamples(lsi) = 0.toByte
            }
          }
          qi += 1
        }
        (siTotalMin, lcActSamples, lcSumCount, lcSumTarget, lcSumSquare, lcSumWeight)

      }

    }
    val actSamples = partAct.map(x => (x._1, x._2)).collect()
    actSamples.par.foreach { case (siMin, lcAct) =>
      Array.copy(lcAct, 0, act, siMin, lcAct.length)
    }
    partAct.map(x => (x._3, x._4, x._5, x._6)).reduce((a, b) => (a._1 + b._1, a._2 + b._2, a._3 + b._3, a._4 + b._4))
  }


  def evaluateErrors(pdcRDD: RDD[(Int, Int)],
                     dcBc: Broadcast[DerivativeCalculator],
                     currentScores: Array[Double],
                     numQueries: Int): Double = {
    val sc = pdcRDD.context
    val currentScoresBc = sc.broadcast(currentScores)
    val sumErrors = pdcRDD.mapPartitions { iter =>
      val dc = dcBc.value
      val currentScores = currentScoresBc.value
      iter.map { case (qiMin, lcNumQueries) =>
        dc.getPartErrors(currentScores, qiMin, qiMin + lcNumQueries)
      }
    }.sum()
    currentScoresBc.unpersist(blocking = false)
    sumErrors / numQueries
  }

  def IsSeleted(ffraction: Double): Boolean = {
    val randomNum = scala.util.Random.nextDouble()
    var active = false
    if (randomNum < ffraction) {
      active = true
    }
    active
  }

}

