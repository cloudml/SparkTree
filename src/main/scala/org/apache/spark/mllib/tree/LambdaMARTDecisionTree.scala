package org.apache.spark.mllib.tree

import breeze.collection.mutable.SparseArray
import org.apache.spark.Logging
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.tree.config.Algo._
import org.apache.spark.mllib.tree.config.Strategy
import org.apache.spark.mllib.tree.impl._
import org.apache.spark.mllib.tree.model.informationgainstats.InformationGainStats
import org.apache.spark.mllib.tree.model.node._
import org.apache.spark.mllib.tree.model.opdtmodel.OptimizedDecisionTreeModel
import org.apache.spark.mllib.tree.model.predict.Predict
import org.apache.spark.mllib.tree.model.{Histogram, NodeInfoStats, SplitInfo}
import org.apache.spark.mllib.util.ProbabilityFunctions
import org.apache.spark.rdd.RDD

import scala.collection.immutable.BitSet
import scala.collection.mutable


class LambdaMARTDecisionTree(val strategy: Strategy,
                             val phaseMinInstancesPerNode: Int,
                             val numLeaves: Int,
                             val maxSplits: Int,
                             val expandTreeEnsemble: Boolean) extends Serializable with Logging {

  strategy.assertValid()

  var curLeaves = 0

  def run(trainingData: RDD[(Int, SparseArray[Short], Array[SplitInfo])],
          trainingData_T: RDD[(Int, Array[Array[Short]])],
          lambdasBc: Broadcast[Array[Double]],
          weightsBc: Broadcast[Array[Double]],
          numSamples: Int,
          entropyCoefft: Double,
          featureUseCount: Array[Int],
          featureFirstUsePenalty: Double,
          featureReusePenalty: Double,
          feature2Gain: Array[Double],
          sampleWeights: Array[Double],
          numPruningLeaves: Int,
          topInfoValue: (Int, Double, Double, Double),
          actSamples: Array[Byte] = Array.empty[Byte]) = {

    val timer = new TimeTracker()

    timer.start("total")

    // depth of the decision tree
    val maxDepth = strategy.maxDepth
    require(maxDepth <= 30, s"LambdaMART currently only supports maxDepth <= 30, but $maxDepth was given.")
    // val maxMemoryUsage: Long = strategy.maxMemoryInMB * 1024L * 1024L
    val minInstancesPerNode = phaseMinInstancesPerNode

    // FIFO queue of nodes to train: node, descending order
    implicit val nodeOrd = Ordering.by[(Node, NodeInfoStats), Double](_._1.impurity)
    // val highQ = MinMaxPriorityQueue.orderedBy(nodeOrd).maximumSize(numLeaves).create[Node]()
    val nodeQueue = new mutable.PriorityQueue[(Node, NodeInfoStats)]()(nodeOrd)
    val topNode = Node.emptyNode(nodeIndex = 1)

    curLeaves = 1

    // Create node Id tracker.
    // At first, all the samples belong to the root nodes (node Id == 1).

    val nodeIdTracker = Array.fill[Int](numSamples)(1)
    val nodeNoTracker = new Array[Byte](numSamples)
    //    val activeSamples = new Array[Byte](numSamples)
    // TBD re-declared
    val nodeId2Score = new mutable.HashMap[Int, Double]()

    //    val topInfo = new NodeInfoStats(numSamples, lambdasBc.value.sum, lambdasBc.value.map(x => x * x).sum, weightsBc.value.sum)

    //    val activeSamples:Array[Byte] = if(actSamples.isEmpty) Array.fill[Byte](numSamples)(1) else actSamples

    val topInfo = new NodeInfoStats(topInfoValue._1, topInfoValue._2, topInfoValue._3, topInfoValue._4)
    nodeQueue.enqueue((topNode, topInfo))

    while (nodeQueue.nonEmpty && (numLeaves == 0 || curLeaves < numLeaves)) {
      val (nodesToSplits, nodesInfo, nodeId2NodeNo) = selectNodesToSplit(nodeQueue, maxSplits)

      Range(0, numSamples).par.foreach(si =>
        nodeNoTracker(si) = nodeId2NodeNo.getOrElse(nodeIdTracker(si), -1)
        //        nodeNoTracker(si) = nodeId2NodeNo.getOrElse(nodeIdTrackerSampler(si), -1)
      )

      // Choose node splits, and enqueue new nodes as needed.
      timer.start("findBestSplits")
      val maxDepth = if (numLeaves > 0) 32 else strategy.maxDepth
      val newSplits = LambdaMARTDecisionTree.findBestSplits(trainingData, trainingData_T, lambdasBc, weightsBc,
        maxDepth, nodesToSplits, nodesInfo, nodeId2Score, nodeNoTracker, nodeQueue, timer,
        minInstancesPerNode, entropyCoefft, featureUseCount, featureFirstUsePenalty,
        featureReusePenalty, feature2Gain, sampleWeights, strategy.algo)

      newSplits.par.foreach { case (siMin, lcNumSamples, splitIndc, isLeftChild) =>
        var lsi = 0
        while (lsi < lcNumSamples) {
          if (splitIndc(lsi)) {
            val si = lsi + siMin
            val oldNid = nodeIdTracker(si)
            //            val oldNidSampler = nodeIdTrackerSampler(si)
            nodeIdTracker(si) = if (isLeftChild(lsi)) oldNid << 1 else (oldNid << 1) + 1
            //            nodeIdTrackerSampler(si) = if (isLeftChild(lsi)) oldNidSampler << 1 else (oldNidSampler << 1) + 1
          }
          lsi += 1
        }
      }

      timer.stop("findBestSplits")
    }

    while (nodeQueue.nonEmpty) {
      nodeQueue.dequeue()._1.isLeaf = true
    }

    println("print topNode")
    println(s"numDescendants: ${topNode.numDescendants}, curLeaves: $curLeaves")

    val treeScores = new Array[Double](numSamples)


    if (numPruningLeaves > 0) {
      timer.start("pruning")
      val mergedNodes =
        strategy.algo match {
          case LambdaMart => pruneNodesForLambdaMart(topNode, numPruningLeaves, maxDepth)
          case Regression => pruneNodesForRegression(topNode, numPruningLeaves, maxDepth, numSamples)
        }

      println("print topNode after pruning")
      println(s"numDescendants: ${topNode.numDescendants}")

      Range(0, numSamples).par.foreach { si =>
        while (mergedNodes(nodeIdTracker(si)) == 1) {
          nodeIdTracker(si) = Node.parentIndex(nodeIdTracker(si))
        }
        treeScores(si) = nodeId2Score(nodeIdTracker(si))
      }
      timer.stop("pruning")
    }
    else {

      Range(0, numSamples).par.foreach { si =>
        treeScores(si) = nodeId2Score(nodeIdTracker(si))
      }
    }
    timer.stop("total")

    println("Internal timing for LambdaMARTDecisionTree:")
    println(s"$timer")


    val model = new OptimizedDecisionTreeModel(topNode, strategy.algo, expandTreeEnsemble)
    (model, treeScores)
  }

  def selectNodesToSplit(nodeQueue: mutable.PriorityQueue[(Node, NodeInfoStats)],
                         maxSplits: Int): (Array[Node], Array[NodeInfoStats], Map[Int, Byte]) = {
    val mutableNodes = new mutable.ArrayBuffer[Node]()
    val mutableNodesInfo = new mutable.ArrayBuffer[NodeInfoStats]()
    val mutableNodeId2NodeNo = new mutable.HashMap[Int, Byte]()
    var numNodes = 0
    while (nodeQueue.nonEmpty && numNodes < maxSplits && (numLeaves == 0 || curLeaves < numLeaves)) {
      val (node, info) = nodeQueue.dequeue()
      mutableNodes += node
      mutableNodesInfo += info
      mutableNodeId2NodeNo(node.id) = numNodes.toByte
      // Check if enough memory remains to add this node to the group.
      // val nodeMemUsage = aggregateSizeForNode() * 8L
      // if (memUsage + nodeMemUsage <= maxMemoryUsage) {
      //   nodeQueue.dequeue()
      //   mutableNodes += node
      //   mutableNodeId2NodeNo(node.id) = numNodes.toByte
      // }
      // memUsage += nodeMemUsage
      curLeaves += 1
      numNodes += 1
    }
    assert(mutableNodes.nonEmpty, s"LambdaMARTDecisionTree selected empty nodes. Error for unknown reason.")
    // Convert mutable maps to immutable ones.
    (mutableNodes.toArray, mutableNodesInfo.toArray, mutableNodeId2NodeNo.toMap)
  }

  def pruneNodesForLambdaMart(topNode: Node, numLeaves: Int, maxDepth: Int): Array[Int] = {
    val id2Node = new mutable.HashMap[Int, Node]()
    val mergedNodes = Array.fill((math.pow(2, maxDepth + 1) + 1).toInt)(0)
    val leafCandidates = new mutable.PriorityQueue[(Int, Double)]()(
      // ascending order
      Ordering.by((_: (Int, Double))._2).reverse
    )
    var cntLeaves = 0
    var cntNodes = 0
    val nodeIterator = topNode.subtreeIterator
    while (nodeIterator.hasNext) {
      cntNodes += 1
      val curNode = nodeIterator.next()
      id2Node(curNode.id) = curNode
      if (!curNode.isLeaf) {
        //        println(s"${curNode.id}, ${curNode.stats.get.gain}")
        if (curNode.leftNode.get.isLeaf && curNode.rightNode.get.isLeaf) {
          leafCandidates.enqueue((curNode.id, curNode.stats.get.gain))
        }
      }
      else
        cntLeaves += 1
    }


    println(s"cnt leaves: $cntLeaves, numLeaves: $numLeaves, cntNodes: $cntNodes")

    while (cntLeaves > numLeaves) {
      val newLeafId = leafCandidates.dequeue()._1
      val newLeafNode = id2Node(newLeafId)
      //      println(s"${newLeafNode.id}, ${newLeafNode.stats.get.gain}")
      newLeafNode.isLeaf = true
      cntLeaves -= 1
      mergedNodes(newLeafNode.leftNode.get.id) = 1
      mergedNodes(newLeafNode.rightNode.get.id) = 1
      val parentId = Node.parentIndex(newLeafId)
      if (id2Node(Node.leftChildIndex(parentId)).isLeaf && id2Node(Node.rightChildIndex(parentId)).isLeaf) {
        leafCandidates.enqueue((parentId, id2Node(parentId).stats.get.gain))
      }
    }

    mergedNodes.toArray
  }

  def pruneNodesForRegression(topNode: Node, numLeaves: Int, maxDepth: Int, numSamples: Int): Array[Int] = {
    val id2Node = new mutable.HashMap[Int, Node]()
    val mergedNodes = Array.fill((math.pow(2, maxDepth + 1) + 1).toInt)(0)
    val leafCandidates = new mutable.PriorityQueue[(Int, Double)]()(
      // ascending order
      Ordering.by((_: (Int, Double))._2).reverse
    )
    var cntLeaves = 0
    var cntNodes = 0
    val nodeIterator = topNode.subtreeIterator
    while (nodeIterator.hasNext) {
      cntNodes += 1
      val curNode = nodeIterator.next()
      id2Node(curNode.id) = curNode
      if (!curNode.isLeaf) {
        //        println(s"${curNode.id}, ${curNode.stats.get.gain}")
        if (curNode.leftNode.get.isLeaf && curNode.rightNode.get.isLeaf) {
          leafCandidates.enqueue((curNode.id, curNode.stats.get.gain))
        }
      }
      else
        cntLeaves += 1
    }


    println(s"cnt leaves: $cntLeaves, numLeaves: $numLeaves, cntNodes: $cntNodes")

    while (cntLeaves > numLeaves) {
      val newLeafId = leafCandidates.dequeue()._1
      val newLeafNode = id2Node(newLeafId)
      //      println(s"${newLeafNode.id}, ${newLeafNode.stats.get.gain}")
      newLeafNode.isLeaf = true
      cntLeaves -= 1
      mergedNodes(newLeafNode.leftNode.get.id) = 1
      mergedNodes(newLeafNode.rightNode.get.id) = 1
      val parentId = Node.parentIndex(newLeafId)
      if (id2Node(Node.leftChildIndex(parentId)).isLeaf && id2Node(Node.rightChildIndex(parentId)).isLeaf) {
        leafCandidates.enqueue((parentId, id2Node(parentId).stats.get.gain))
      }
    }

    mergedNodes.toArray
  }
}

object LambdaMARTDecisionTree extends Serializable with Logging {
  def findBestSplits(trainingData: RDD[(Int, SparseArray[Short], Array[SplitInfo])],
                     trainingData_T: RDD[(Int, Array[Array[Short]])],
                     lambdasBc: Broadcast[Array[Double]],
                     weightsBc: Broadcast[Array[Double]],
                     maxDepth: Int,
                     nodesToSplit: Array[Node],
                     nodesInfo: Array[NodeInfoStats],
                     nodeId2Score: mutable.HashMap[Int, Double],
                     nodeNoTracker: Array[Byte],
                     nodeQueue: mutable.PriorityQueue[(Node, NodeInfoStats)],
                     timer: TimeTracker,
                     minDocPerNode: Int,
                     entropyCoefft: Double,
                     featureUseCount: Array[Int],
                     featureFirstUsePenalty: Double,
                     featureReusePenalty: Double,
                     feature2Gain: Array[Double],
                     sampleWeights: Array[Double],
                     algo: Algo,
                     activeSamples: Array[Byte] = Array.empty): Array[(Int, Int, BitSet, BitSet)] = {
    // numNodes:  Number of nodes in this group
    val numNodes = nodesToSplit.length
    logDebug("numNodes = " + numNodes)

    // Calculate best splits for all nodes in the group
    timer.start("chooseSplits")
    val sc = trainingData.sparkContext
    val nodeNoTrackerBc = sc.broadcast(nodeNoTracker)
    val nodesInfoBc = sc.broadcast(nodesInfo)
//    val activeSamplesBc = sc.broadcast(activeSamples)

    val bestSplitsPerFeature = trainingData.mapPartitions { iter =>
      val lcNodeNoTracker = nodeNoTrackerBc.value
      val lcLambdas = lambdasBc.value
      val lcWeights = weightsBc.value
      val lcNodesInfo = nodesInfoBc.value
//      val lcActSample = activeSamplesBc.value
      iter.map { case (_, sparseSamples, splits) =>
        val numBins = splits.length
        val histograms = Array.fill(numNodes)(new Histogram(numBins))

        var offset = 0
        while (offset < sparseSamples.activeSize) {
          val index: Int = sparseSamples.indexAt(offset)
          val value: Short = sparseSamples.valueAt(offset)
          val ni = lcNodeNoTracker(index)
          val sampleWeight = if (sampleWeights == null) 1.0 else sampleWeights(index)
          //          if (lcActSample(index)==1 && ni >= 0) {
          //            histograms(ni).update(value, sampleWeight, lcLambdas(index), lcWeights(index))
          //          }
          if (ni >= 0) {
            histograms(ni).update(value, sampleWeight, lcLambdas(index), lcWeights(index))
//            histograms(ni).update(value, sampleWeight, lcLambdas(index),0.0)
          }
          offset += 1
        }
        algo match {
          case LambdaMart =>
            //            val layerSplit=false
            //            if(layerSplit){
            //               binsToBestSplitLayer(histograms, sparseSamples.default, splits, lcNodesInfo, entropyCoefft, featureUseCount,
            //                featureFirstUsePenalty, featureReusePenalty, minInstancesPerNode = minDocPerNode)
            //            }
            //            else{
            Array.tabulate(numNodes)(ni => binsToBestSplitForLambdaMart(histograms(ni), sparseSamples.default, splits, lcNodesInfo(ni), entropyCoefft, featureUseCount,
              featureFirstUsePenalty, featureReusePenalty, minInstancesPerNode = minDocPerNode))
          //            }
          case Regression =>
            Array.tabulate(numNodes)(ni => binsToBestSplitForClassfication(histograms(ni), sparseSamples.default, splits, lcNodesInfo(ni)))
        }
      }
    }

    val bsf = betterSplits(numNodes) _
    val bestSplits = bestSplitsPerFeature.reduce(bsf)
    timer.stop("chooseSplits")

    // Iterate over all nodes in this group.
    var sni = 0
    while (sni < numNodes) {
      val node = nodesToSplit(sni)
      val nodeId = node.id
      val ((split, bin), stats, gainPValue, leftNodeInfo, rtNodeInfo) = bestSplits(sni)
      logDebug(s"best split = $split, bin = $bin")
      featureUseCount(split.feature) += 1 //TODO
      // Extract info for this node.  Create children if not leaf.
      val isLeaf = (stats.gain <= 0) || (Node.indexToLevel(nodeId) == maxDepth)

      node.isLeaf = isLeaf
      node.stats = Some(stats)
      node.impurity = stats.impurity
      logDebug("Node = " + node)
      logDebug("LeftInfo = " + leftNodeInfo)
      logDebug("RightInfo = " + rtNodeInfo)
      nodeId2Score(node.id) = node.predict.predict
      //nodePredict.predict(node.id, nodeIdTrackerBc, targetScoresBc, weightsBc)

      if (!isLeaf) {
        feature2Gain(split.feature) += stats.gain
        node.split = Some(split)
        val childIsLeaf = (Node.indexToLevel(nodeId) + 1) == maxDepth
        //
        val leftChildIsLeaf = childIsLeaf || (stats.leftImpurity <= 0.01)
        val rightChildIsLeaf = childIsLeaf || (stats.rightImpurity <= 0.01)
        node.leftNode = Some(Node(Node.leftChildIndex(nodeId),
          stats.leftPredict, stats.leftImpurity, leftChildIsLeaf))
        nodeId2Score(node.leftNode.get.id) = node.leftNode.get.predict.predict

        node.rightNode = Some(Node(Node.rightChildIndex(nodeId),
          stats.rightPredict, stats.rightImpurity, rightChildIsLeaf))
        nodeId2Score(node.rightNode.get.id) = node.rightNode.get.predict.predict

        // enqueue left child and right child if they are not leaves
        if (!leftChildIsLeaf) {
          nodeQueue.enqueue((node.leftNode.get, leftNodeInfo))
        }
        if (!rightChildIsLeaf) {
          nodeQueue.enqueue((node.rightNode.get, rtNodeInfo))
        }

        logDebug(s"leftChildIndex = ${node.leftNode.get.id}, impurity = ${stats.leftImpurity}")
        logDebug(s"rightChildIndex = ${node.rightNode.get.id}, impurity = ${stats.rightImpurity}")
      }
      sni += 1
    }

    val bestSplitsBc = sc.broadcast(bestSplits)
    val newNodesToSplitBc = sc.broadcast(nodesToSplit)
    val newSplits = trainingData_T.mapPartitions { iter =>
      val lcNodeNoTracker = nodeNoTrackerBc.value
      val lcBestSplits = bestSplitsBc.value
      val lcNewNodesToSplit = newNodesToSplitBc.value
      val (siMin, sampleSlice) = iter.next()
      val lcNumSamples = sampleSlice(0).length
      val splitIndc = new mutable.BitSet(lcNumSamples)
      val isLeftChild = new mutable.BitSet(lcNumSamples)
      var lsi = 0
      while (lsi < lcNumSamples) {
        val oldNi = lcNodeNoTracker(lsi + siMin)
        if (oldNi >= 0) {
          val node = lcNewNodesToSplit(oldNi)
          if (!node.isLeaf) {
            splitIndc += lsi
            val split = lcBestSplits(oldNi)._1._1
            val bin = lcBestSplits(oldNi)._1._2
            if (sampleSlice(split.feature)(lsi) <= bin) {
              isLeftChild += lsi
            }
          }
        }
        lsi += 1
      }
      Iterator.single((siMin, lcNumSamples, splitIndc.toImmutable, isLeftChild.toImmutable))
    }.collect()

    bestSplitsBc.unpersist(blocking = false)
    newNodesToSplitBc.unpersist(blocking = false)
    nodeNoTrackerBc.unpersist(blocking = false)
    nodesInfoBc.unpersist(blocking = false)
    newSplits
  }

  def betterSplits(numNodes: Int)(a: Array[((SplitInfo, Short), InformationGainStats, Double, NodeInfoStats, NodeInfoStats)],
                                  b: Array[((SplitInfo, Short), InformationGainStats, Double, NodeInfoStats, NodeInfoStats)])
  : Array[((SplitInfo, Short), InformationGainStats, Double, NodeInfoStats, NodeInfoStats)] = {
    Array.tabulate(numNodes) { ni =>
      val ai = a(ni)
      val bi = b(ni)
      if (ai._2.gain >= bi._2.gain) ai else bi
    }
  }

  // TODO: make minInfoGain parameterized
  // for lambdaMart
  def binsToBestSplitForLambdaMart(hist: Histogram, defaultBin: Int,
                                   splits: Array[SplitInfo],
                                   nodeInfo: NodeInfoStats,
                                   entropyCoefft: Double,
                                   featureUseCount: Array[Int],
                                   featureFirstUsePenalty: Double,
                                   featureReusePenalty: Double,
                                   minInstancesPerNode: Int = 1,
                                   minGain: Double = Double.MinPositiveValue): ((SplitInfo, Short), InformationGainStats, Double, NodeInfoStats, NodeInfoStats) = {
    /**
      * val counts = hist.counts
      * val sumTargets = hist.scores
      * val sumWeights = hist.scoreWeights **/

    val feature = splits(0).feature

    val cumHist = hist.cumulate(nodeInfo, defaultBin)

    //val denom = if (sumWeight == 0.0) totalDocInNode else sumWeight
    val varianceTargets = (nodeInfo.sumSquares - nodeInfo.sumScores / nodeInfo.sumCount) / (nodeInfo.sumCount - 1)
    //val eps = 1e-10
    val gainShift = getLeafSplitGain(nodeInfo.sumCount, nodeInfo.sumScores) //TODO use only weight or secondDrivatives
    // TODO
    val gainConfidenceLevel = 0.0
    var gainConfidenceInSquaredStandardDeviations = ProbabilityFunctions.Probit(1.0 - (1.0 - gainConfidenceLevel) * 0.5)
    gainConfidenceInSquaredStandardDeviations *= gainConfidenceInSquaredStandardDeviations

    val minShiftedGain = if (gainConfidenceInSquaredStandardDeviations <= 0) 0.0 //TODO
    else (gainConfidenceInSquaredStandardDeviations * varianceTargets
      * nodeInfo.sumCount / (nodeInfo.sumCount - 1) + gainShift)

    val entropyCoefficient = entropyCoefft * 1.0e-6 //TODO
    val bestRtInfo = new NodeInfoStats(-1, Double.NaN, Double.NaN, Double.NaN)
    var bestShiftedGain = Double.NegativeInfinity
    var bestThreshold = 0.0
    var bestThresholdBin = 0


    var i = 1
    while (i < splits.length) {
      val threshLeft = i
      val rtCount = cumHist.counts(threshLeft).toInt
      val lteCount = nodeInfo.sumCount - rtCount
      val rtSumTarget = cumHist.scores(threshLeft)
       val rtSumWeight = cumHist.scoreWeights(threshLeft)
      val lteSumTarget = nodeInfo.sumScores - rtSumTarget
       val lteSumWeight = nodeInfo.sumScoreWeights - rtSumWeight
      val th = i - 1

      // val gain = lscores * lscores / lcnts + rscores * rscores / rcnts  gainShift >= minShiftedGain
      if (lteCount >= minInstancesPerNode && rtCount >= minInstancesPerNode) {
        var currentShiftedGain = getLeafSplitGain(lteCount, lteSumTarget) + getLeafSplitGain(rtCount, rtSumTarget)

        if (currentShiftedGain > minShiftedGain) {
          //TODO
          if (entropyCoefficient > 0) {
            //TODO
            val entropyGain = nodeInfo.sumCount * math.log(nodeInfo.sumCount) - lteCount * math.log(lteCount) -
              rtCount * math.log(rtCount)
            currentShiftedGain += entropyCoefficient * entropyGain
          }

          if (currentShiftedGain > bestShiftedGain) {
            bestRtInfo.sumCount = rtCount
            bestRtInfo.sumScores = rtSumTarget
            bestRtInfo.sumSquares = cumHist.squares(threshLeft)
            bestRtInfo.sumScoreWeights = cumHist.scoreWeights(threshLeft)
            bestShiftedGain = currentShiftedGain
            bestThreshold = splits(threshLeft - 1).threshold
            bestThresholdBin = i - 1
          }
        }
      }
      i += 1
    }
    //    val gtSquares = sumSquares - bestLteSquaredTarget
    //    val gtTarget = sumTargets - bestLteTarget
    //    val gtCount = totalDocInNode - bestLteCount
    val bestLeftInfo = new NodeInfoStats(nodeInfo.sumCount - bestRtInfo.sumCount, nodeInfo.sumScores - bestRtInfo.sumScores,
      nodeInfo.sumSquares - bestRtInfo.sumSquares, nodeInfo.sumScoreWeights - bestRtInfo.sumScoreWeights)

    val lteImpurity = (bestLeftInfo.sumSquares - bestLeftInfo.sumScores * bestLeftInfo.sumScores / bestLeftInfo.sumCount) / bestLeftInfo.sumCount
    val gtImpurity = (bestRtInfo.sumSquares - bestRtInfo.sumScores * bestRtInfo.sumScores / bestRtInfo.sumCount) / bestRtInfo.sumCount
    val tolImpurity = (nodeInfo.sumSquares - nodeInfo.sumScores * nodeInfo.sumScores / nodeInfo.sumCount) / nodeInfo.sumCount

    val bestSplitInfo = (new SplitInfo(feature, bestThreshold), bestThresholdBin.toShort)
    val lteOutput = CalculateSplittedLeafOutput(bestLeftInfo.sumCount, bestLeftInfo.sumScores, bestLeftInfo.sumScoreWeights)
    val gtOutput = CalculateSplittedLeafOutput(bestRtInfo.sumCount, bestRtInfo.sumScores, bestRtInfo.sumScoreWeights)
    val ltePredict = new Predict(lteOutput)
    val gtPredict = new Predict(gtOutput)

    val trust = 1.0 // TODO
    //println("#############################################################################################")
    //println(s"bestShiftedGain: $bestShiftedGain, gainShift: $gainShift")
    val usePenalty = if (featureUseCount(feature) == 0) featureFirstUsePenalty //TODO
      else featureReusePenalty * scala.math.log(featureUseCount(feature) + 1)

    val splitGain = (bestShiftedGain - gainShift) * trust - usePenalty //TODO introduce trust and usePenalty


    val inforGainStat = new InformationGainStats(splitGain, tolImpurity, lteImpurity, gtImpurity, ltePredict, gtPredict)
    val erfcArg = math.sqrt((bestShiftedGain - gainShift) * (nodeInfo.sumCount - 1) / (2 * varianceTargets * nodeInfo.sumCount))
    val gainPValue = ProbabilityFunctions.erfc(erfcArg)

    (bestSplitInfo, inforGainStat, gainPValue, bestLeftInfo, bestRtInfo)
  }

  // for classification
  def binsToBestSplitForClassfication(hist: Histogram, defaultBin: Int,
                                      splits: Array[SplitInfo],
                                      nodeInfo: NodeInfoStats,
                                      minInstancesPerNode: Int = 1): ((SplitInfo, Short), InformationGainStats, Double, NodeInfoStats, NodeInfoStats) = {
    /**
      * val counts = hist.counts
      * val sumTargets = hist.scores
      * val sumWeights = hist.scoreWeights **/

    val feature = splits(0).feature
    val cumHist = hist.cumulate(nodeInfo, defaultBin)

    //gain=impurity-leftWeight*leftImpurity-rightWeight*rightImpurity

    val impurity = (nodeInfo.sumSquares - nodeInfo.sumScores * nodeInfo.sumScores / nodeInfo.sumCount) / nodeInfo.sumCount

    val bestRtInfo = new NodeInfoStats(-1, Double.NaN, Double.NaN, 0.0)
    var bestGain = Double.NegativeInfinity
    var bestThreshold = 0.0
    var bestThresholdBin = 0
    var leftImpurity = 0.0
    var rightImpurity = 0.0

    var i = 1
    while (i < splits.length) {
      val threshLeft = i
      val rtCount = cumHist.counts(threshLeft).toInt
      val lteCount = nodeInfo.sumCount - rtCount
      val rtSumTarget = cumHist.scores(threshLeft)
      val lteSumTarget = nodeInfo.sumScores - rtSumTarget
      val rtSumSquareTarget = cumHist.squares(threshLeft)
      val lteSumSquareTarget = nodeInfo.sumSquares - rtSumSquareTarget
      val th = i - 1

      if (lteCount >= minInstancesPerNode && rtCount >= minInstancesPerNode) {
        val lftImpurity = getImpurity(lteCount, lteSumTarget, lteSumSquareTarget)
        val rtImpurity = getImpurity(rtCount, rtSumTarget, rtSumSquareTarget)
        val lftWeight = lteCount / nodeInfo.sumCount.toDouble
        val rtWeight = rtCount / nodeInfo.sumCount.toDouble
        val currentGain = impurity - lftImpurity * lftWeight - rtImpurity * rtWeight

        if (currentGain > Double.MinPositiveValue) {
          //TODO

          if (currentGain > bestGain) {
            bestRtInfo.sumCount = rtCount
            bestRtInfo.sumScores = rtSumTarget
            bestRtInfo.sumSquares = cumHist.squares(threshLeft)
            bestGain = currentGain
            bestThreshold = splits(threshLeft - 1).threshold
            bestThresholdBin = i - 1
            leftImpurity = lftImpurity
            rightImpurity = rtImpurity
          }
        }
      }
      i += 1
    }
    val bestLeftInfo = new NodeInfoStats(nodeInfo.sumCount - bestRtInfo.sumCount, nodeInfo.sumScores - bestRtInfo.sumScores,
      nodeInfo.sumSquares - bestRtInfo.sumSquares, 0.0)

    val bestSplitInfo = (new SplitInfo(feature, bestThreshold), bestThresholdBin.toShort)
    val lteOutput = getPredict(bestLeftInfo.sumCount, bestLeftInfo.sumScores)
    val gtOutput = getPredict(bestRtInfo.sumCount, bestRtInfo.sumScores)
    val ltePredict = new Predict(lteOutput)
    val gtPredict = new Predict(gtOutput)


    val inforGainStat = new InformationGainStats(bestGain*nodeInfo.sumCount, impurity, leftImpurity, rightImpurity, ltePredict, gtPredict)
    val erfcArg = math.sqrt(bestGain * (nodeInfo.sumCount - 1) / (2 * impurity * nodeInfo.sumCount))
    val gainPValue = ProbabilityFunctions.erfc(erfcArg)

    (bestSplitInfo, inforGainStat, gainPValue, bestLeftInfo, bestRtInfo)
  }

  // for lambdaMart
  def binsToBestSplitLayer(hists: Array[Histogram], defaultBin: Int,
                           splits: Array[SplitInfo],
                           nodeInfos: Array[NodeInfoStats],
                           entropyCoefft: Double,
                           featureUseCount: Array[Int],
                           featureFirstUsePenalty: Double,
                           featureReusePenalty: Double,
                           minInstancesPerNode: Int = 1,
                           minGain: Double = Double.MinPositiveValue): Array[((SplitInfo, Short), InformationGainStats, Double, NodeInfoStats, NodeInfoStats)] = {

    val feature = splits(0).feature
    val numNodes = hists.length

    val cumHists = Array.tabulate(numNodes) { ni => hists(ni).cumulate(nodeInfos(ni), defaultBin) }

    val gainConfidenceLevel = 0.0
    var gainConfidenceInSquaredStandardDeviations = ProbabilityFunctions.Probit(1.0 - (1.0 - gainConfidenceLevel) * 0.5)
    gainConfidenceInSquaredStandardDeviations *= gainConfidenceInSquaredStandardDeviations

    val minShiftedGain = minGain
    val gainShifts = Array.tabulate(numNodes)(ni => getLeafSplitGain(nodeInfos(ni).sumCount, nodeInfos(ni).sumScores))

    val bestRtInfos = Array.fill(numNodes)(new NodeInfoStats(-1, Double.NaN, Double.NaN, Double.NaN))
    var bestLayerGain = Double.NegativeInfinity
    var bestThreshold = 0.0
    var bestThresholdBin = 0

    var i = 1
    while (i < splits.length) {
      val threshLeft = i
      var layerGain = 0.0
      val rtCounts = new Array[Int](numNodes)
      val lteCounts = new Array[Int](numNodes)
      val rtSumTargets = new Array[Double](numNodes)
      val lteSumTargets = new Array[Double](numNodes)

      Range(0, numNodes).foreach { ni =>
        rtCounts(ni) = cumHists(ni).counts(threshLeft).toInt
        lteCounts(ni) = nodeInfos(ni).sumCount - rtCounts(ni)
        rtSumTargets(ni) = cumHists(ni).scores(threshLeft)
        lteSumTargets(ni) = nodeInfos(ni).sumScores - rtSumTargets(ni)
        val th = i - 1

        if (lteCounts(ni) >= minInstancesPerNode && rtCounts(ni) >= minInstancesPerNode) {
          layerGain += getLeafSplitGain(lteCounts(ni), lteSumTargets(ni)) + getLeafSplitGain(rtCounts(ni), rtSumTargets(ni)) - gainShifts(ni)
        }
      }

      if (layerGain > bestLayerGain) {
        Range(0, numNodes).foreach { ni =>
          bestRtInfos(ni) = new NodeInfoStats(rtCounts(ni), rtSumTargets(ni), cumHists(ni).squares(threshLeft), 0.0)
        }
        bestThreshold = splits(threshLeft - 1).threshold
        bestThresholdBin = i - 1
        bestLayerGain = layerGain
      }
      i += 1
    }

    Array.tabulate(numNodes) { ni =>
      val bestLeftInfo = new NodeInfoStats(nodeInfos(ni).sumCount - bestRtInfos(ni).sumCount, nodeInfos(ni).sumScores - bestRtInfos(ni).sumScores,
        nodeInfos(ni).sumSquares - bestRtInfos(ni).sumSquares, nodeInfos(ni).sumScoreWeights - bestRtInfos(ni).sumScoreWeights)
      val lteImpurity = (bestLeftInfo.sumSquares - bestLeftInfo.sumScores * bestLeftInfo.sumScores / bestLeftInfo.sumCount) / bestLeftInfo.sumCount
      val gtImpurity = (bestRtInfos(ni).sumSquares - bestRtInfos(ni).sumScores * bestRtInfos(ni).sumScores / bestRtInfos(ni).sumCount) / bestRtInfos(ni).sumCount
      val tolImpurity = (nodeInfos(ni).sumSquares - nodeInfos(ni).sumScores * nodeInfos(ni).sumScores / nodeInfos(ni).sumCount) / nodeInfos(ni).sumCount

      val bestSplitInfo = (new SplitInfo(feature, bestThreshold), bestThresholdBin.toShort)
      val lteOutput = CalculateSplittedLeafOutput(bestLeftInfo.sumCount, bestLeftInfo.sumScores, bestLeftInfo.sumScoreWeights)
      val gtOutput = CalculateSplittedLeafOutput(bestRtInfos(ni).sumCount, bestRtInfos(ni).sumScores, bestRtInfos(ni).sumScoreWeights)
      val ltePredict = new Predict(lteOutput)
      val gtPredict = new Predict(gtOutput)

      val inforGainStat = new InformationGainStats(bestLayerGain, tolImpurity, lteImpurity, gtImpurity, ltePredict, gtPredict)
      val gainPValue = 0.0
      (bestSplitInfo, inforGainStat, gainPValue, bestLeftInfo, bestRtInfos(ni))
    }
  }

  def getLeafSplitGain(count: Double, target: Double): Double = {
    //val pesuedCount = if(weight == 0.0) count else weight
    target * target / count
  }

  def getImpurity(count: Double, target: Double, squareTarget: Double): Double = {
    if (count == 0) 0
    else
      (squareTarget - target * target * 1.0 / count) / count
  }

  def getPredict(count: Double, target: Double): Double = {
    if (count == 0) 0
    else target * 1.0 / count
  }

  def CalculateSplittedLeafOutput(totalCount: Int, sumTargets: Double, sumWeights: Double): Double = {
    //    val hasWeight = false
    //    val bsrMaxTreeOutput = 100.0
    //    if (!hasWeight) {
    //      //TODO hasweight true or false
    //      sumTargets / totalCount
    //    } else {
    //      if (bsrMaxTreeOutput < 0.0) {
    //        //TODO  bsrMaxTreeOutput default 100
    //        sumTargets / sumWeights
    //      } else {
    //        sumTargets / (2 * sumWeights)
    //      }
    //    }
    val epsilon = 1.4e-45
    val maxOutput = 100
    val rawOutput = sumTargets / totalCount
    val mean = sumWeights / totalCount

    var leafValue = (rawOutput + epsilon) / (2 * mean + epsilon)

    if (leafValue > maxOutput)
      leafValue = maxOutput
    if (leafValue < -maxOutput)
      leafValue = -maxOutput

    leafValue
  }
}
