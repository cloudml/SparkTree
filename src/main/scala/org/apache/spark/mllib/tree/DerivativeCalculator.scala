package org.apache.spark.mllib.tree

import scala.collection.mutable.ArrayBuffer

class DerivativeCalculator extends Serializable {
  val expAsymptote: Double = -50
  val sigmoidBins: Int = 1000000

  var sigmoidTable: Array[Double] = null
  var minScore: Double = _
  var maxScore: Double = _
  var scoreToSigmoidTableFactor: Double =_
  var _distanceWeight2 = false  //TODO CMD PARAMETER

  var discounts: Array[Double] = null
  var ratings: Array[Short] = null
  var gainTable: Array[Double] = null

  val _normalizeQueryLambdas = true

  val maxNumPositions = 5000
  // var labels: Array[Byte]
  var secondaryGains: Array[Double] = null
  var _secondaryMetricShare: Double =_
  var _secondaryIsolabelExclusive: Boolean=_
  var _secondaryInverseMaxDCGT: Array[Double] = null

  var queryBoundy: Array[Int] = null
  var inverseMaxDCGs: Array[Double] = null
  var _baselineAlpha: Array[Double] = null
  var _baselineDcg: Array[Double] = null

  def init(ratings: Array[Short], gainTable: Array[Double], queryBoundy: Array[Int], sigma: Double,
           distanceWeight2: Boolean, baselineAlpha: Array[Double],
           secondaryMetricShare: Double,
           secondaryIsolabelExclusive: Boolean,
           secondaryGain: Array[Double],
           secondaryInverseMaxDcg: Array[Double],
           discount: Array[Double],
           baselineDcgs: Array[Double]): Unit = {
    initSigmoidTable(sigma)
    _distanceWeight2 = distanceWeight2
    _baselineAlpha = baselineAlpha


    _baselineDcg = baselineDcgs
    discounts = if(discount == null) {
      Array.tabulate(maxNumPositions)(i => 1.0 / math.log(i + 2.0))
    }
    else{
      discount
    }

    this.ratings = ratings
    this.gainTable = gainTable

    setupSecondaryGains(secondaryMetricShare,
      secondaryIsolabelExclusive,
      secondaryGain,
      secondaryInverseMaxDcg)

    calcInverseMaxDCGs(queryBoundy)
  }

  private def initSigmoidTable(sigma: Double): Unit = {
    // minScore is such that 2*sigma*score is < expAsymptote if score < minScore
    minScore = expAsymptote / sigma / 2
    maxScore = -minScore
    scoreToSigmoidTableFactor = sigmoidBins / (maxScore - minScore)

    sigmoidTable = new Array[Double](sigmoidBins)
    var i = 0
    while (i < sigmoidBins) {
      val score = (maxScore - minScore) / sigmoidBins * i + minScore
      sigmoidTable(i) = if (score > 0.0) {
        2.0 - 2.0 / (1.0 + math.exp(-2.0 * sigma * score))
      } else {
        2.0 / (1.0 + math.exp(2.0 * sigma * score))
      }
      i += 1
    }

  }

  def setupSecondaryGains(secondaryMetricShare: Double,
                          secondaryIsolabelExclusive: Boolean,
                          secondaryGain: Array[Double],
                          secondaryInverseMaxDcg: Array[Double]): Unit ={
    _secondaryMetricShare = secondaryMetricShare
    _secondaryIsolabelExclusive = secondaryIsolabelExclusive
    secondaryGains = secondaryGain

    if(secondaryMetricShare != 0.0 && secondaryGains != null){
      _secondaryInverseMaxDCGT = secondaryInverseMaxDcg
    }
  }

  private def calcInverseMaxDCGs(queryBoundy: Array[Int]): Unit = {
    this.queryBoundy = queryBoundy
    val numQueries = queryBoundy.length - 1
    inverseMaxDCGs = new Array[Double](numQueries)
    var qi = 0
    while (qi < numQueries) {
      val siMin = queryBoundy(qi)
      val siEnd = queryBoundy(qi + 1)
      val ratings_sorted = ratings.view(siMin, siEnd).toSeq.sorted.reverse.toArray

      var MaxDCGQ = 0.0
      val numDocs = siEnd - siMin
      var odi = 0
      while (odi < numDocs) {
        MaxDCGQ += gainTable(ratings_sorted(odi)) * discounts(odi)
        odi += 1
      }
      val inverseMaxDCGQ = if (MaxDCGQ == 0.0) 0.0 else 1.0 / MaxDCGQ
      inverseMaxDCGs(qi) = inverseMaxDCGQ
      //println(">>>>>>>>>>>>>>")
      //println(s"query: $qi, numdocs: $numDocs")
      //println(ratings.view(siMin, siEnd).mkString("\t"))
      //println(s"MaxDcg: $MaxDCGQ")
      qi += 1
    }
  }

  private def ScoreSort(scores: Array[Double], siMin: Int, siEnd: Int): Array[Short] = {
    scores.view(siMin, siEnd).map(_.toShort).toSeq.sorted.reverse.toArray
  }

  private def docIdxSort(scores: Array[Double], siMin: Int, siEnd: Int): Array[Int] = {
    Range(siMin, siEnd).sortBy(scores).reverse.map(_ - siMin).toArray
  }

  def getPartDerivatives(scores: Array[Double], qiMin: Int, qiEnd: Int, iteration: Int): (Int, Array[Double], Array[Double]) = {
    val siTotalMin = queryBoundy(qiMin)
    val numTotalDocs = queryBoundy(qiEnd) - siTotalMin
    val lcLambdas = new Array[Double](numTotalDocs)
    val lcWeights = new Array[Double](numTotalDocs)
    var qi = qiMin
    while (qi < qiEnd) {
      val lcMin = queryBoundy(qi) - siTotalMin

      val siMin = queryBoundy(qi)
      val siEnd = queryBoundy(qi + 1)
      val numDocsPerQuery = siEnd - siMin
      val permutation = docIdxSort(scores, siMin, siEnd)

      var baselineVersusCurrentDcg = 0.0
      if(_baselineDcg != null){
        baselineVersusCurrentDcg = _baselineDcg(qi)
        for(i<- 0 until numDocsPerQuery){
          baselineVersusCurrentDcg -= gainTable(ratings(permutation(i) + siMin)) * discounts(i)
        }
      }

      val baselineAlphaRisk = if(_baselineAlpha == null) 0.0 else _baselineAlpha(iteration)
      val secondaryIMDcg = if(_secondaryInverseMaxDCGT == null) 1.0 else _secondaryInverseMaxDCGT(qi)

      calcQueryDerivatives(qi, scores, lcLambdas, lcWeights, permutation, lcMin,
        _secondaryMetricShare, _secondaryIsolabelExclusive, _distanceWeight2,
        baselineAlphaRisk, secondaryIMDcg, baselineVersusCurrentDcg)
      qi += 1
    }
    (siTotalMin, lcLambdas, lcWeights)
  }

      def getPartErrors(scores: Array[Double], qiMin: Int, qiEnd: Int): Double = {
       var errors = 0.0
       var qi = qiMin
       while (qi < qiEnd) {
          val siMin = queryBoundy(qi)
          val siEnd = queryBoundy(qi + 1)
          val numDocs = siEnd - siMin
          val permutation = docIdxSort(scores, siMin, siEnd)
          var dcg = 0.0
          var odi = 0
          while (odi < numDocs) {
              dcg += gainTable(ratings(permutation(odi) + siMin)) * discounts(odi)
              odi += 1
          }
          errors += 1 - dcg * inverseMaxDCGs(qi)
          qi += 1
       }
       errors
      }

  private def calcQueryDerivatives(qi: Int,
                                   scores: Array[Double],
                                   lcLambdas: Array[Double],
                                   lcWeights: Array[Double],
                                   permutation: Array[Int],
                                   lcMin: Int,
                                   secondaryMetricShare: Double,
                                   secondaryExclusive: Boolean,
                                   distanceWeight2: Boolean,
                                   alphaRisk: Double,
                                   secondaryInverseMaxDCG: Double,
                                   baselineVersusCurrentDcg: Double,
                                   costFunctionParam: Char = 'w',
                                   minDoubleValue: Double = Double.MinValue): Unit = {
    val siMin = queryBoundy(qi)
    val siEnd = queryBoundy(qi + 1)
    val numDocs = siEnd - siMin
    val tmpArray = new ArrayBuffer[Double]
    for(i <- 0 until numDocs){
      tmpArray += scores(i + siMin)
    }

    val inverseMaxDCG = inverseMaxDCGs(qi)
    /**
      * println(">>>>>>>>>>>>>>")
      * println(s"query: $qi, numdocs: $numDocs")
      * println(s"label: " + ratings.view(siMin, siEnd).mkString(",") + "\t" + s"permutation: " + permutation.mkString(","))
      * println(s"scores: " + scores.view(siMin, siEnd).mkString(",") + "\t" + s"discount: " + discounts.view(0,20).mkString(","))
      * println(s"inverseMaxDcg: $inverseMaxDCG")
      * println(s"mins: $minScore, maxs: $maxScore, factor: $scoreToSigmoidTableFactor")
      **
      *
      *println("**************")
      *println(tmpArray.toString())
      *println(permutation.mkString(","))
      **
      *println(s"inverseMaxDCG: $inverseMaxDCG")  **/

    val bestScore = scores(permutation.head + siMin)
    var worstIndexToConsider = numDocs - 1
    while (worstIndexToConsider > 0 && scores(permutation(worstIndexToConsider) + siMin) == minDoubleValue) {
      worstIndexToConsider -= 1
    }
    val worstScore = scores(permutation(worstIndexToConsider) + siMin)

    var lambdaSum = 0.0

    // Should we still run the calculation on those pairs which are ostensibly the same?
    val pairSame = secondaryMetricShare != 0.0


    // Did not help to use pointer match on pPermutation[i]
    for (odi <- 0 until numDocs) {
      val di = permutation(odi)
      val sHigh = di + siMin
      val labelHigh =ratings(sHigh)
      val scoreHigh = scores(sHigh)

      if (!((labelHigh == 0 && !pairSame) || scoreHigh == minDoubleValue)) {
        var deltaLambdasHigh: Double = 0.0
        var deltaWeightsHigh: Double = 0.0

        for (odj <- 0 until numDocs) {

          val dj = permutation(odj)
          val sLow = dj + siMin
          val labelLow = ratings(sLow)
          val scoreLow = scores(sLow)

          val flag = if (pairSame) labelHigh < labelLow else labelHigh <= labelLow
          if (!(flag || scores(sLow) == minDoubleValue)) {
            val scoreHighMinusLow = scoreHigh - scoreLow
            if (!(secondaryMetricShare == 0.0 && labelHigh == labelLow && scoreHighMinusLow <= 0)) {

              //println("labelHigh", labelHigh, "aLabels(siLow)", aLabels(siLow), "scoreHighMinusLow", scoreHighMinusLow)
              var dcgGap: Double = gainTable(ratings(sHigh)) - gainTable(ratings(sLow))
              var currentInverseMaxDCG = inverseMaxDCG * (1.0 - secondaryMetricShare)

              val pairedDiscount = (discounts(odi) - discounts(odj)).abs
              if (alphaRisk > 0) {
                var  risk = 0.0
                val baselineDenorm = baselineVersusCurrentDcg / pairedDiscount
                if (baselineVersusCurrentDcg > 0) {
                  risk = if (scoreHighMinusLow <= 0 && dcgGap > baselineDenorm) baselineDenorm else dcgGap
                } else if (scoreHighMinusLow > 0) {
                  // The baseline is currently lower, but this pair is ranked correctly.
                  risk = baselineDenorm + dcgGap
                }
                if (risk > 0) {
                  dcgGap += alphaRisk * risk
                }
              }

              val lambdaP = if (scoreHighMinusLow <= minScore) {
                sigmoidTable.head
              } else if (scoreHighMinusLow >= maxScore) {
                sigmoidTable.last
              } else {
                sigmoidTable(((scoreHighMinusLow - minScore) * scoreToSigmoidTableFactor).toInt)
              }
              val weightP = lambdaP * (2.0 - lambdaP)

              var sameLabel = labelHigh == labelLow
              if (!(secondaryMetricShare != 0.0 && (sameLabel || currentInverseMaxDCG == 0.0) && secondaryGains != null &&secondaryGains(sHigh) <= secondaryGains(sLow))) {
                if (secondaryMetricShare != 0.0) {
                  if (sameLabel || currentInverseMaxDCG == 0.0) {
                    // We should use the secondary metric this time.
                    dcgGap = secondaryGains(sHigh) - secondaryGains(sLow)
                    currentInverseMaxDCG = secondaryInverseMaxDCG * secondaryMetricShare
                    sameLabel = false
                  } else if (!secondaryExclusive && secondaryGains(sHigh) > secondaryGains(sLow)) {
                    var sIDCG = secondaryInverseMaxDCG * secondaryMetricShare
                    dcgGap = dcgGap / sIDCG + (secondaryGains(sHigh) - secondaryGains(sLow)) / currentInverseMaxDCG
                    currentInverseMaxDCG *= sIDCG
                  }
                }
                // calculate the deltaNDCGP for this pair
                var deltaNDCGP = dcgGap * pairedDiscount * currentInverseMaxDCG

                // apply distanceWeight2 only to regular pairs
                if (!sameLabel && distanceWeight2 && bestScore != worstScore) {
                  deltaNDCGP /= (.01 + (scoreHigh - scoreLow).abs)
                }
                //println("lambda", lambdaP * deltaNDCGP, "deltaNDCGP", deltaNDCGP, "dcgGap", dcgGap, "pairedDiscount", pairedDiscount, "currentInverseMaxDCG", currentInverseMaxDCG)
                // update lambdas and weights
                deltaLambdasHigh += lambdaP * deltaNDCGP
                // println("*****************")
                //println(s"lambdaP: $lambdaP, deltaNDCGP: $deltaNDCGP")
                deltaWeightsHigh += weightP * deltaNDCGP
                lcLambdas(dj + lcMin) -= lambdaP * deltaNDCGP
                lcWeights(dj + lcMin) += weightP * deltaNDCGP

                lambdaSum += 2 * lambdaP * deltaNDCGP
              }
            }
          }
        }
        //Finally, add the values for the siHigh part of the pair that we accumulated across all the low parts

        lcLambdas(di + lcMin) += deltaLambdasHigh
        lcWeights(di + lcMin) += deltaWeightsHigh
      }
    }

    /***
      *if(qi < 15)
      *{
      *println(s"lambdas in query $qi")
      *for(i <- 0 until numDocs)
      *{
      *print(lcLambdas(lcMin + i) + ",")
      *}
      *println()
      *} ***/

    if(_normalizeQueryLambdas)
    {
      if(lambdaSum > 0)
      {
        val normFactor = (10 * math.log(1 + lambdaSum))/lambdaSum
        for(i <- 0 until numDocs)
        {
          lcLambdas(lcMin + i) = lcLambdas(lcMin + i) * normFactor
          lcWeights(lcMin + i) = lcWeights(lcMin + i) * normFactor
        }
      }
    }

  }
}

/*****
  *object Derivate {
  *def main(args: Array[String]){
  *val numDocuments = 5; val begin = 0
  *val aPermutation = Array(1, 4, 3, 4, 2); val aLabels: Array[Short] = Array(1, 2, 3, 4, 5)
  *val aScores = Array(1.0, 3.0, 8.0, 15.0, 31.0)
  *val aDiscount = Array(0.2, 0.5, 0.7, 0.8, 0.9)
  *val inverseMaxDCG = 0.01
  *val aGainLabels = Array(0.3, 0.4, 0.5, 0.8, 0.3)
  *val aSecondaryGains = Array(0.3, 0.4, 0.5, 0.8, 0.3); val asigmoidTable =GetDerivatives.FillSigmoidTable()
  *val minScore = 0.08; val maxScore = 0.2
  *val scoreToSigmoidTableFactor = 4
  **
  *GetDerivatives.GetDerivatives_lambda_weight(
  *numDocuments, begin,
  *aPermutation, aLabels,
  *aScores,
  *aDiscount, aGainLabels, inverseMaxDCG,
  *asigmoidTable, minScore, maxScore, scoreToSigmoidTableFactor, aSecondaryGains
  *)
  * }
*}
  *****/
