package org.apache.spark.mllib.tree.model

class Histogram(val numBins: Int) {
  private val _counts = new Array[Double](numBins)
  private val _scores = new Array[Double](numBins)
  private val _squares = new Array[Double](numBins)
  private val _scoreWeights = new Array[Double](numBins)

  @inline def counts = _counts

  @inline def scores = _scores

  @inline def squares = _squares

  @inline def scoreWeights = _scoreWeights

  def weightedUpdate(bin: Int, score: Double, scoreWeight: Double, weight: Double = 1.0) = {
    _counts(bin) += weight
    _scores(bin) += score * weight
    _squares(bin) += score * score * weight
    _scoreWeights(bin) += scoreWeight
  }
  def update(bin: Int, sampleWeight: Double, score: Double, scoreWeight: Double) = {
    _counts(bin) += sampleWeight
    _scores(bin) += score
    _squares(bin) += score * score
    _scoreWeights(bin) += scoreWeight
  }

  def cumulateLeft() = {
    var bin = 1
    while (bin < numBins) {
      _counts(bin) += _counts(bin-1)
      _scores(bin) += _scores(bin-1)
      _squares(bin) += _squares(bin-1)
      _scoreWeights(bin) += _scoreWeights(bin-1)
      bin += 1
    }
    this
  }

  def cumulate(info: NodeInfoStats, defaultBin: Int)={
    // cumulate from right to left
    var bin = numBins-2
    var binRight = 0
    while (bin > defaultBin) {
      binRight = bin+1
      _counts(bin) += _counts(binRight)
      _scores(bin) += _scores(binRight)
      _squares(bin) += _squares(binRight)
      _scoreWeights(bin) += _scoreWeights(binRight)
      bin -= 1
    }

    if(defaultBin!=0){
      _counts(0)=info.sumCount-_counts(0)
      _scores(0)=info.sumScores-_scores(0)
      _squares(0)=info.sumSquares-_squares(0)
      _scoreWeights(0)=info.sumScoreWeights-_scoreWeights(0)

      bin = 1
      var binLeft = 0
      while(bin<defaultBin){
        _counts(bin)=_counts(binLeft)-_counts(bin)
        _scores(bin)=_scores(binLeft)-_scores(bin)
        _squares(bin)=_squares(binLeft)-_squares(bin)
        _scoreWeights(bin)=_scoreWeights(binLeft)-_scoreWeights(bin)
        bin+=1
        binLeft+=1
      }
      bin = defaultBin
      binLeft = defaultBin-1
      while(bin>0){
        _counts(bin)=_counts(binLeft)
        _scores(bin)=_scores(binLeft)
        _squares(bin)=_squares(binLeft)
        _scoreWeights(bin)=_scoreWeights(binLeft)
        bin-=1
        binLeft-=1
      }
    }
    _counts(0)=info.sumCount
    _scores(0)=info.sumScores
    _squares(0)=info.sumSquares
    _scoreWeights(0)=info.sumScoreWeights

    this

  }
}

class NodeInfoStats(var sumCount: Int,
                    var sumScores: Double,
                    var sumSquares: Double,
                    var sumScoreWeights: Double)extends Serializable {

  override def toString = s"NodeInfoStats( sumCount = $sumCount, sumTarget = $sumScores, sumSquares = $sumSquares, sumScoreWeight = $sumScoreWeights)"

  def canEqual(other: Any): Boolean = other.isInstanceOf[NodeInfoStats]

  override def equals(other: Any): Boolean = other match {
    case that: NodeInfoStats =>
      (that canEqual this) &&
        sumCount == that.sumCount &&
        sumScores == that.sumScores &&
        sumSquares == that.sumSquares &&
        sumScoreWeights == that.sumScoreWeights
    case _ => false
  }

  override def hashCode(): Int = {
    val state = Seq(sumCount, sumScores, sumSquares, sumScoreWeights)
    state.map(_.hashCode()).foldLeft(0)((a, b) => 31 * a + b)
  }
}
