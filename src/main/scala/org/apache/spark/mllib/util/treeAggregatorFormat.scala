package org.apache.spark.mllib.util

import java.io.{File, FileOutputStream, PrintWriter}
import java.text.SimpleDateFormat
import java.util.Date

//import org.apache.spark.ml.param.Param

import org.apache.spark.examples.mllib.LambdaMARTRunner.Params

import scala.util.Sorting


object treeAggregatorFormat{
  def appendTreeAggregator(expandTreeEnsemble: Boolean,
    filePath: String,
    index: Int,
    evalNodes: Array[Int],
    evalWeights: Array[Double] = null,
    bias: Double = 0.0,
    Type: String = "Linear"): Unit = {
    val pw = new PrintWriter(new FileOutputStream(new File(filePath), true))

    pw.append(s"[Evaluator:$index]").write("\r\n")
    pw.append(s"EvaluatorType=Aggregator").write("\r\n")

    val numNodes = if(expandTreeEnsemble) evalNodes.length+1 else evalNodes.length
    val defaultWeight = 1.0
    if (evalNodes == null) {
      throw new IllegalArgumentException("there is no evaluators to be aggregated")
    } else {
      pw.append(s"NumNodes=$numNodes").write("\r\n")
      pw.append(s"Nodes=").write("")

      if(expandTreeEnsemble) {
        pw.append(s"I:1").write("\t")
      }
      for (eval <- evalNodes) {
        pw.append(s"E:$eval").write("\t")
      }
      pw.write("\r\n")
    }

    var weights = new Array[Double](numNodes)
    if (evalWeights == null) {
      for (i <- 0 until numNodes) {
        weights(i) = defaultWeight
      }
    } else {
      weights = evalWeights
    }

    pw.append(s"Weights=").write("")
    for (weight <- weights) {
      pw.append(s"$weight").write("\t")
    }

    pw.write("\r\n")

    pw.append(s"Bias=$bias").write("\r\n")
    pw.append(s"Type=$Type").write("\r\n")
    pw.close()
  }

 // format comment
  def toCommentFormat(filePath: String, param: Params, featureToName: Array[String],
                      featureToGain: Array[Double]): Unit ={
    val pw = new PrintWriter(new FileOutputStream(new File(filePath), true))

    val gainSorted = featureToGain.zipWithIndex
    Sorting.quickSort(gainSorted)(Ordering.by[(Double, Int), Double](_._1).reverse)

    if(param.GainNormalization){
      val normFactor = gainSorted(0)._1
      gainSorted.map{case (gain, idx) =>
        (gain/normFactor, idx)}
    }

    val timeFormat = new SimpleDateFormat("dd/MM/yyyy HH:mm:ss")
    val currentTime = timeFormat.format(new Date())
    pw.append(s"\n[Comments]\nC:0=Regression Tree Ensemble\nC:1=Generated using spark FastRank\nC:2=Created on $currentTime\n").write("")

    var skip = 3
    val paramsList = param.toString.split("\n")
    val NumParams = paramsList.length
    for(i <- 0 until NumParams){
      pw.append(s"C:${skip + i}=PARAM:${paramsList(i)}\n").write("")
    }

    val offset = if(param.expandTreeEnsemble) 2 else 1
    skip += NumParams
    val NumFeatures = gainSorted.length

    for(i <- 0 until NumFeatures){
      var substr = featureToName(gainSorted(i)._2)
      if(substr.length > 68)
          substr = substr.substring(0, 67) + "..."
      pw.append(s"C:${skip + i}=FG:I${gainSorted(i)._2 + offset}:$substr:${gainSorted(i)._1}\n").write("")
    }
 }


}
