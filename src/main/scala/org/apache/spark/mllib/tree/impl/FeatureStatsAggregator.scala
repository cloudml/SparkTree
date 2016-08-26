package org.apache.spark.mllib.tree.impl

import org.apache.spark.mllib.tree.model.impurity._

class FeatureStatsAggregator(
    val numBins: Byte) extends Serializable {
    
  val impurityAggregator: ImpurityAggregator = new VarianceAggregator()
  
  private val statsSize: Int = impurityAggregator.statsSize
  
  private val allStatsSize: Int = numBins * statsSize
  
  private val allStats: Array[Double] = new Array[Double](allStatsSize)
  
  def getImpurityCalculator(binIndex: Int): ImpurityCalculator = {
    impurityAggregator.getCalculator(allStats, binIndex * statsSize)
  }
  
  def update(binIndex: Int, label: Double, instanceWeight: Double, weight: Double): Unit = {
    val i = binIndex * statsSize
    impurityAggregator.update(allStats, i, label, instanceWeight, weight)
  }
  
  def merge(binIndex: Int, otherBinIndex: Int): Unit = {
    impurityAggregator.merge(allStats, binIndex * statsSize, otherBinIndex * statsSize)
  }
}
