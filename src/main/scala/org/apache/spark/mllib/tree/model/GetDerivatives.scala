package org.apache.spark.mllib.tree.model

import scala.math
import scala.collection.mutable
import scala.collection.mutable.HashMap
import scala.collection.mutable.ArrayBuilder

object GetDerivatives {

  val _expAsymptote: Double = -50
  val _sigmoidBins: Int = 1000000
  var _minScore: Double = 0.0
  var _maxScore: Double = 0.0
 // var _sigmoidTable = _
  var _scoreToSigmoidTableFactor: Double = 0.0
  var _minSigmoid: Double = 0.0
  var _maxSigmoid: Double = 0.0

  def sortArray(arr: Array[Double], st: Int, num: Int): Array[Int] = {
    val score2idx = scala.collection.mutable.HashMap.empty[Double, Int]
    for(i <- 0 until  num)
      score2idx(arr(st + i)) = i

    var res = new Array[Int](num)
    var i = 0
    var listSorted = score2idx.toList.sorted
    listSorted.foreach { case (key, value) =>
      res(i) = value
      i += 1
    }
    res
  }

  def labelSort(arr: Array[Double], st: Int, num: Int): Array[Double] = {
    var res = new Array[Double](num)
    for(i <- 0 until num){
      res(i) = arr(st + i)
    }
    res.sortWith(_ > _)
    res
  }

  def FillSigmoidTable(sigmoidParam: Double = 1.0): Array[Double]= {
    // minScore is such that 2*sigmoidParam*score is < expAsymptote if score < minScore
    _minScore = _expAsymptote / sigmoidParam / 2
    _maxScore = -_minScore

    var _sigmoidTable = new Array[Double](_sigmoidBins)
    for (i <- 0 until _sigmoidBins) {
      var score: Double = (_maxScore - _minScore) / _sigmoidBins * i + _minScore

      _sigmoidTable(i) =
        if (score > 0.0)
          2.0 - 2.0 / (1.0 + scala.math.exp(-2.0 * sigmoidParam * score))
        else
          2.0 / (1.0 + scala.math.exp(2.0 * sigmoidParam * score))
    }
    _scoreToSigmoidTableFactor = _sigmoidBins / (_maxScore - _minScore)
    _minSigmoid = _sigmoidTable(0)
    _maxSigmoid = _sigmoidTable.last
    _sigmoidTable
  }

  def GetDerivatives_lambda_weight(
                                    numDocuments: Int, begin: Int,
                                    aPermutation: Array[Int], aLabels: Array[Short],
                                    aScores: Array[Double], aLambdas: Array[Double], aWeights: Array[Double],
                                    aDiscount: Array[Double], aGainLabels: Array[Double], inverseMaxDCG: Double,
                                    asigmoidTable: Array[Double],  minScore: Double, maxScore: Double, scoreToSigmoidTableFactor: Double,
                                    aSecondaryGains: Array[Double], secondaryMetricShare: Double = 0.0, secondaryExclusive: Boolean = false, secondaryInverseMaxDCG: Double = 0.2,
                                    costFunctionParam: Char = 'c', distanceWeight2: Boolean = false,  minDoubleValue: Double = 0.01,
                                    alphaRisk: Double = 0.2, baselineVersusCurrentDcg: Double = 0.1) {
    // These arrays are shared among many threads, "begin" is the offset by which all arrays are indexed.
    //  So we shift them all here to avoid having to add 'begin' to every pointer below.
    //val pLabels = begin
    //val pScores = begin
    //val pLambdas = begin
    //val pWeights = begin
    //val pGainLabels = begin
    //println("here0")
    //var aLambdas = new Array[Double](aLabels.length)
    //var aWeights = new Array[Double](aLabels.length)

    var pSecondaryGains = 0

    if (secondaryMetricShare != 0)
      pSecondaryGains = begin

    var bestScore = aScores(aPermutation(0))

    var worstIndexToConsider = numDocuments - 1

    while (worstIndexToConsider > 0 && aScores(aPermutation(worstIndexToConsider)) == minDoubleValue) {
      worstIndexToConsider -= 1
    }
    var worstScore = aScores(aPermutation(worstIndexToConsider))

    var lambdaSum = 0.0

    // Should we still run the calculation on those pairs which are ostensibly the same?
    var pairSame: Boolean = secondaryMetricShare != 0.0

    // Did not help to use pointer match on pPermutation[i]
    for (i <- 0 until numDocuments)
    {
      //println("here1")
      var high = begin + aPermutation(i)
      // We are going to loop through all pairs where label[high] > label[low]. If label[high] is 0, it can't be larger
      // If score[high] is Double.MinValue, it's being discarded by shifted NDCG
      //println("aLabels(high)", aLabels(high), "aScores(high)", aScores(high), "minDoubleValue", minDoubleValue, "pairSame", pairSame)
      if (!((aLabels(high) == 0 && !pairSame) || aScores(high) == minDoubleValue)) { // These variables are all looked up just once per loop of 'i', so do it here.
        
        var gainLabelHigh = aGainLabels(high)
        var labelHigh = aLabels(high)
        var scoreHigh = aScores(high)
        var discountI = aDiscount(i)
        // These variables will store the accumulated lambda and weight difference for high, which saves time
        var deltaLambdasHigh: Double = 0
        var deltaWeightsHigh: Double = 0

        //The below is effectively: for (int j = 0; j < numDocuments; ++j)
        var aaDiscountJ: Array[Double] = aDiscount
        var aaPermutationJ: Array[Int] = aPermutation

        for (j <- 0 until numDocuments) {
          // only consider pairs with different labels, where "high" has a higher label than "low"
          // If score[low] is Double.MinValue, it's being discarded by shifted NDCG
          var low = begin + aaPermutationJ(j)
          var flag =
            if (pairSame) labelHigh < aLabels(low)
            else
              labelHigh <= aLabels(low)
          if (!(flag || aScores(low) == minDoubleValue)) {
            var scoreHighMinusLow = scoreHigh - aScores(low)
            if (!(secondaryMetricShare == 0.0 && labelHigh == aLabels(low) && scoreHighMinusLow <= 0)) {

              //println("labelHigh", labelHigh, "aLabels(low)", aLabels(low), "scoreHighMinusLow", scoreHighMinusLow)
              var dcgGap = gainLabelHigh - aGainLabels(low)
              var currentInverseMaxDCG = inverseMaxDCG * (1.0 - secondaryMetricShare)

              // Handle risk w.r.t. baseline.
              var pairedDiscount = (discountI - aaDiscountJ(j)).abs
              if (alphaRisk > 0) {
                var risk: Double = 0.0
                var baselineDenorm: Double = baselineVersusCurrentDcg / pairedDiscount
                if (baselineVersusCurrentDcg > 0) {
                  // The baseline is currently higher than the model.
                  // If we're ranked incorrectly, we can only reduce risk only as much as the baseline current DCG.
                  risk =
                    if (scoreHighMinusLow <= 0 && dcgGap > baselineDenorm) baselineDenorm
                    else
                      dcgGap
                } else if (scoreHighMinusLow > 0) {
                  // The baseline is currently lower, but this pair is ranked correctly.
                  risk = baselineDenorm + dcgGap
                }
                if (risk > 0) {
                  dcgGap += alphaRisk * risk
                }
              }

              var sameLabel: Boolean = labelHigh == aLabels(low)

              // calculate the lambdaP for this pair by looking it up in the lambdaTable (computed in LambdaMart.FillLambdaTable)
              var lambdaP = 0.0
              if (scoreHighMinusLow <= minScore)
                lambdaP = asigmoidTable(0)
              else if (scoreHighMinusLow >= maxScore) lambdaP = asigmoidTable(asigmoidTable.length - 1)
              else lambdaP = asigmoidTable(((scoreHighMinusLow - minScore) * scoreToSigmoidTableFactor).toInt)

              
              var weightP = lambdaP * (2.0 - lambdaP)

              if (!(secondaryMetricShare != 0.0 && (sameLabel || currentInverseMaxDCG == 0.0) && aSecondaryGains(high) <= aSecondaryGains(low))) {
                if (secondaryMetricShare != 0.0) {
                  if (sameLabel || currentInverseMaxDCG == 0.0) {
                    // We should use the secondary metric this time.
                    dcgGap = aSecondaryGains(high) - aSecondaryGains(low)
                    currentInverseMaxDCG = secondaryInverseMaxDCG * secondaryMetricShare
                    sameLabel = false
                  } else if (!secondaryExclusive && aSecondaryGains(high) > aSecondaryGains(low)) {
                    var sIDCG = secondaryInverseMaxDCG * secondaryMetricShare
                    dcgGap = dcgGap / sIDCG + (aSecondaryGains(high) - aSecondaryGains(low)) / currentInverseMaxDCG
                    currentInverseMaxDCG *= sIDCG
                  }
                }
                //println("here2")
                //printf("%d-%d : gap %g, currentinv %g\n", high, low, (float)dcgGap, (float)currentInverseMaxDCG); fflush(stdout);

                // calculate the deltaNDCGP for this pair
                var deltaNDCGP = dcgGap * pairedDiscount * currentInverseMaxDCG
                
                // apply distanceWeight2 only to regular pairs
                if (!sameLabel && distanceWeight2 && bestScore != worstScore) {
                  deltaNDCGP /= (.01 + (aScores(high) - aScores(low)).abs)
                }
                //println("lambda", lambdaP * deltaNDCGP, "deltaNDCGP", deltaNDCGP, "dcgGap", dcgGap, "pairedDiscount", pairedDiscount, "currentInverseMaxDCG", currentInverseMaxDCG)
                // update lambdas and weights
                deltaLambdasHigh += lambdaP * deltaNDCGP
                deltaWeightsHigh += weightP * deltaNDCGP
                aLambdas(low) -= lambdaP * deltaNDCGP
                aWeights(low) += weightP * deltaNDCGP

                lambdaSum += 2 * lambdaP * deltaNDCGP
              }
            }
          }
        }
        //Finally, add the values for the high part of the pair that we accumulated across all the low parts

        aLambdas(high) += deltaLambdasHigh
        aWeights(high) += deltaWeightsHigh 
        
        //for(i <- 0 until numDocuments) println(aLambdas(begin + i), aWeights(begin + i))
      }
    }
    (aLambdas, aWeights)
  }
}


/*****
object Derivate {
  def main(args: Array[String]){
    val numDocuments = 5; val begin = 0
    val aPermutation = Array(1, 4, 3, 4, 2); val aLabels: Array[Short] = Array(1, 2, 3, 4, 5)
    val aScores = Array(1.0, 3.0, 8.0, 15.0, 31.0)
    val aDiscount = Array(0.2, 0.5, 0.7, 0.8, 0.9)
    val inverseMaxDCG = 0.01
    val aGainLabels = Array(0.3, 0.4, 0.5, 0.8, 0.3)
    val aSecondaryGains = Array(0.3, 0.4, 0.5, 0.8, 0.3); val asigmoidTable =GetDerivatives.FillSigmoidTable()
    val minScore = 0.08; val maxScore = 0.2
    val scoreToSigmoidTableFactor = 4

    GetDerivatives.GetDerivatives_lambda_weight(
      numDocuments, begin,
      aPermutation, aLabels,
      aScores,
      aDiscount, aGainLabels, inverseMaxDCG,
      asigmoidTable, minScore, maxScore, scoreToSigmoidTableFactor, aSecondaryGains
    )
  }
}
*****/