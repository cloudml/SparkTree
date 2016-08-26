/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.mllib.tree.model.opdtmodel

import java.io.{File, FileOutputStream, PrintWriter}

import org.apache.spark.annotation.Experimental
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.tree.config.Algo
import org.apache.spark.mllib.tree.config.Algo._
import org.apache.spark.mllib.tree.configuration.FeatureType
import org.apache.spark.mllib.tree.model.Split
import org.apache.spark.mllib.tree.model.informationgainstats.InformationGainStats
import org.apache.spark.mllib.tree.model.node.Node
import org.apache.spark.mllib.tree.model.predict.Predict
import org.apache.spark.mllib.util.{Loader, Saveable}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.util.Utils
import org.apache.spark.{Logging, SparkContext}
import org.json4s.JsonDSL._
import org.json4s._
import org.json4s.jackson.JsonMethods._

import scala.collection.mutable

/**
 * :: Experimental ::
 * Decision tree model for classification or regression.
 * This model stores the decision tree structure and parameters.
 * @param topNode root node
 * @param algo algorithm type -- classification or regression
 */
@Experimental
class OptimizedDecisionTreeModel(val topNode: Node, val algo: Algo, val expandTreeEnsemble: Boolean = false)
  extends Serializable with Saveable {
  type Lists = (List[String], List[Double], List[Double], List[Int], List[Int], List[Double], List[Double])

  /**
   * Predict values for a single data point using the model trained.
   *
   * @param features array representing a single data point
   * @return Double prediction from the trained model
   */
  def predict(features: Vector): Double = {
    topNode.predict(features)
  }

  /**
   * Predict values for the given data set using the model trained.
   *
   * @param features RDD representing data points to be predicted
   * @return RDD of predictions for each of the given data points
   */
  def predict(features: RDD[Vector]): RDD[Double] = {
    features.map(x => predict(x))
  }

  /**
   * Predict values for the given data set using the model trained.
   *
   * @param features JavaRDD representing data points to be predicted
   * @return JavaRDD of predictions for each of the given data points
   */
  def predict(features: JavaRDD[Vector]): JavaRDD[Double] = {
    predict(features.rdd)
  }

  /**
   * Get number of nodes in tree, including leaf nodes.
   */
  def numNodes: Int = 10
  // {
  //   1 + topNode.numDescendants
  // }

  /**
   * Get depth of tree.
   * E.g.: Depth 0 means 1 leaf node.  Depth 1 means 1 internal node and 2 leaf nodes.
   */
  def depth: Int = {
    5
    // topNode.subtreeDepth
  }

  // def internalNodes(rootNode: Node): Int = {
  //    if(rootNode.isLeaf == true){

  //    }
  //   internalNodes(rootNode.leftNode)+internalNodes(rootNode.rightNode)+1
  // }

  /**
   * Print a summary of the model.
   */
  override def toString: String = algo match {
    case Classification =>
      s"OptimizedDecisionTreeModel classifier  with $numNodes leaf nodes"
    case Regression =>
      s"OptimizedDecisionTreeModel regressor  with $numNodes leaf nodes"
    case _ => throw new IllegalArgumentException(
      s"OptimizedDecisionTreeModel given unknown algo parameter: $algo.")
  }

  /**
   * Print the full model to a string.
   */
  def toDebugString: String = {
    val header = toString + "\n"
    header + topNode.subtreeToString(2)
  }

  override def save(sc: SparkContext, path: String): Unit = {
    OptimizedDecisionTreeModel.SaveLoadV1_0.save(sc, path, this)
  }

  def reformatted: Lists = {
    val splitFeatures = new mutable.MutableList[String]
    val splitGains = new mutable.MutableList[Double]
    val gainPValues = new mutable.MutableList[Double]
    val lteChildren = new mutable.MutableList[Int]
    val gtChildren = new mutable.MutableList[Int]
    val thresholds = new mutable.MutableList[Double]
    val outputs = new mutable.MutableList[Double]

    var curNonLeafIdx = 0
    var curLeafIdx = 0
    val childIdx = (child: Node) => if (child.isLeaf) {
      curLeafIdx -= 1
      curLeafIdx
    } else {
      curNonLeafIdx += 1
      curNonLeafIdx
    }

    val q = new mutable.Queue[Node]
    q.enqueue(topNode)
    while (q.nonEmpty) {
      val node = q.dequeue()
      if (!node.isLeaf) {
        val split = node.split.get
        val stats = node.stats.get

        val offSet =  if(expandTreeEnsemble) 2 else 1
        splitFeatures += s"I:${split.feature+offSet}"

        splitGains += stats.gain
        gainPValues += 0.0
        thresholds += split.threshold
        val left = node.leftNode.get
        val right = node.rightNode.get
        lteChildren += childIdx(left)
        gtChildren += childIdx(right)
        q.enqueue(left)
        q.enqueue(right)
      } else {
        outputs += node.predict.predict
      }
    }
    (splitFeatures.toList, splitGains.toList, gainPValues.toList, lteChildren.toList, gtChildren.toList,
      thresholds.toList, outputs.toList)
  }

  def sequence(path: String, model: OptimizedDecisionTreeModel, modelId: Int): Unit = {
    val (splitFeatures, splitGains, gainPValues, lteChildren, gtChildren, thresholds, outputs) = reformatted

    val pw = new PrintWriter(new FileOutputStream(new File(path), true))
    if(1 == modelId)
        pw.write(s"\n")
    pw.write(s"[Evaluator:$modelId]\n")
    pw.write("EvaluatorType=DecisionTree\n")
    pw.write(s"NumInternalNodes=${topNode.internalNodes}\n")

    var str = splitFeatures.mkString("\t")
    pw.write(s"SplitFeatures=$str\n")
    str = splitGains.mkString("\t")
    pw.write(s"SplitGain=$str\n")
    str = gainPValues.mkString("\t")
    pw.write(s"GainPValue=$str\n")
    str = lteChildren.mkString("\t")
    pw.write(s"LTEChild=$str\n")
    str = gtChildren.mkString("\t")
    pw.write(s"GTChild=$str\n")
    str = thresholds.mkString("\t")
    pw.write(s"Threshold=$str\n")
    str = outputs.mkString("\t")
    pw.write(s"Output=$str\n")

    pw.write("\n")
    pw.close()

  }

  override protected def formatVersion: String = OptimizedDecisionTreeModel.formatVersion
}

object OptimizedDecisionTreeModel extends Loader[OptimizedDecisionTreeModel] with Logging {

    private[spark] def formatVersion: String = "1.0"

    private[tree] object SaveLoadV1_0 {

      def thisFormatVersion: String = "1.0"

      // Hard-code class name string in case it changes in the future
      def thisClassName: String = "org.apache.spark.mllib.tree.OptimizedDecisionTreeModel"

      case class PredictData(predict: Double, prob: Double) {
        def toPredict: Predict = new Predict(predict, prob)
      }

      object PredictData {
        def apply(p: Predict): PredictData = PredictData(p.predict, p.prob)

        def apply(r: Row): PredictData = PredictData(r.getDouble(0), r.getDouble(1))
      }

      case class SplitData(
                            feature: Int,
                            threshold: Double,
                            featureType: Int,
                            categories: Seq[Double]) { // TODO: Change to List once SPARK-3365 is fixed
      def toSplit: Split = {
        new Split(feature, threshold, FeatureType(featureType), categories.toList)
      }
      }

      object SplitData {
        def apply(s: Split): SplitData = {
          SplitData(s.feature, s.threshold, s.featureType.id, s.categories)
        }

        def apply(r: Row): SplitData = {
          SplitData(r.getInt(0), r.getDouble(1), r.getInt(2), r.getAs[Seq[Double]](3))
        }
      }

      /** Model data for model import/export */
      case class TreeNodeData(
                               treeId: Int,
                               nodeId: Int,
                               predict: PredictData,
                               impurity: Double,
                               isLeaf: Boolean,
                               split: Option[SplitData],
                               leftNodeId: Option[Int],
                               rightNodeId: Option[Int],
                               infoGain: Option[Double])

      object TreeNodeData {
        def apply(treeId: Int, n: Node): TreeNodeData = {
          TreeNodeData(treeId, n.id, PredictData(n.predict), n.impurity, n.isLeaf,
            n.split.map(SplitData.apply), n.leftNode.map(_.id), n.rightNode.map(_.id),
            n.stats.map(_.gain))
        }

        def apply(r: Row): TreeNodeData = {
          val split = if (r.isNullAt(5)) None else Some(SplitData(r.getStruct(5)))
          val leftNodeId = if (r.isNullAt(6)) None else Some(r.getInt(6))
          val rightNodeId = if (r.isNullAt(7)) None else Some(r.getInt(7))
          val infoGain = if (r.isNullAt(8)) None else Some(r.getDouble(8))
          TreeNodeData(r.getInt(0), r.getInt(1), PredictData(r.getStruct(2)), r.getDouble(3),
            r.getBoolean(4), split, leftNodeId, rightNodeId, infoGain)
        }
      }



      def save(sc: SparkContext, path: String, model: OptimizedDecisionTreeModel): Unit = {
        val sqlContext = new SQLContext(sc)
        import sqlContext.implicits._

        // SPARK-6120: We do a hacky check here so users understand why save() is failing
        //             when they run the ML guide example.
        // TODO: Fix this issue for real.
        val memThreshold = 768
        if (sc.isLocal) {
          val driverMemory = sc.getConf.getOption("spark.driver.memory")
            .orElse(Option(System.getenv("SPARK_DRIVER_MEMORY")))
            .map(Utils.memoryStringToMb)
            .getOrElse(512)
          if (driverMemory <= memThreshold) {
            logWarning(s"$thisClassName.save() was called, but it may fail because of too little" +
              s" driver memory (${driverMemory}m)." +
              s"  If failure occurs, try setting driver-memory ${memThreshold}m (or larger).")
          }
        } else {
          if (sc.executorMemory <= memThreshold) {
            logWarning(s"$thisClassName.save() was called, but it may fail because of too little" +
              s" executor memory (${sc.executorMemory}m)." +
              s"  If failure occurs try setting executor-memory ${memThreshold}m (or larger).")
          }
        }

        // Create JSON metadata.
        val metadata = compact(render(
          ("class" -> thisClassName) ~ ("version" -> thisFormatVersion) ~
            ("algo" -> model.algo.toString) ~ ("numNodes" -> model.numNodes)))
        sc.parallelize(Seq(metadata), 1).saveAsTextFile(Loader.metadataPath(path))

        // Create Parquet data.
        val nodes = model.topNode.subtreeIterator.toSeq
        val dataRDD: DataFrame = sc.parallelize(nodes)
          .map(TreeNodeData.apply(0, _))
          .toDF()
        dataRDD.write.parquet(Loader.dataPath(path))
      }

      def load(sc: SparkContext, path: String, algo: String, numNodes: Int): OptimizedDecisionTreeModel = {
        val datapath = Loader.dataPath(path)
        val sqlContext = new SQLContext(sc)
        // Load Parquet data.
        val dataRDD = sqlContext.read.parquet(datapath)
        // Check schema explicitly since erasure makes it hard to use match-case for checking.
        Loader.checkSchema[TreeNodeData](dataRDD.schema)
        val nodes = dataRDD.map(TreeNodeData.apply)
        // Build node data into a tree.
        val trees = constructTrees(nodes)
        assert(trees.size == 1,
          "Decision tree should contain exactly one tree but got ${trees.size} trees.")
        val model = new OptimizedDecisionTreeModel(trees(0), Algo.fromString(algo))
        assert(model.numNodes == numNodes, s"Unable to load OptimizedDecisionTreeModel data from: $datapath." +
          s" Expected $numNodes nodes but found ${model.numNodes}")
        model
      }

      def constructTrees(nodes: RDD[TreeNodeData]): Array[Node] = {
        val trees = nodes
          .groupBy(_.treeId)
          .mapValues(_.toArray)
          .collect()
          .map { case (treeId, data) =>
          (treeId, constructTree(data))
        }.sortBy(_._1)
        val numTrees = trees.size
        val treeIndices = trees.map(_._1).toSeq
        assert(treeIndices == (0 until numTrees),
          s"Tree indices must start from 0 and increment by 1, but we found $treeIndices.")
        trees.map(_._2)
      }

      /**
       * Given a list of nodes from a tree, construct the tree.
        *
        * @param data array of all node data in a tree.
       */
      def constructTree(data: Array[TreeNodeData]): Node = {
        val dataMap: Map[Int, TreeNodeData] = data.map(n => n.nodeId -> n).toMap
        assert(dataMap.contains(1),
          s"OptimizedDecisionTree missing root node (id = 1).")
        constructNode(1, dataMap, mutable.Map.empty)
      }

      /**
       * Builds a node from the node data map and adds new nodes to the input nodes map.
       */
      private def constructNode(
                                 id: Int,
                                 dataMap: Map[Int, TreeNodeData],
                                 nodes: mutable.Map[Int, Node]): Node = {
        if (nodes.contains(id)) {
          return nodes(id)
        }
        val data = dataMap(id)
        val node =
          if (data.isLeaf) {
            Node(data.nodeId, data.predict.toPredict, data.impurity, data.isLeaf)
          } else {
            val leftNode = constructNode(data.leftNodeId.get, dataMap, nodes)
            val rightNode = constructNode(data.rightNodeId.get, dataMap, nodes)
            val stats = new InformationGainStats(data.infoGain.get, data.impurity, leftNode.impurity,
              rightNode.impurity, leftNode.predict, rightNode.predict)
            new Node(data.nodeId, data.predict.toPredict, data.impurity, data.isLeaf,
              data.split.map(_.toSplit), Some(leftNode), Some(rightNode), Some(stats))
          }
        nodes += node.id -> node
        node
      }
    }

    override def load(sc: SparkContext, path: String): OptimizedDecisionTreeModel = {
      implicit val formats = DefaultFormats
      val (loadedClassName, version, metadata) = Loader.loadMetadata(sc, path)
      val algo = (metadata \ "algo").extract[String]
      val numNodes = (metadata \ "numNodes").extract[Int]
      val classNameV1_0 = SaveLoadV1_0.thisClassName
      (loadedClassName, version) match {
        case (className, "1.0") if className == classNameV1_0 =>
          SaveLoadV1_0.load(sc, path, algo, numNodes)
        case _ => throw new Exception(
          s"OptimizedDecisionTreeModel.load did not recognize model with (className, format version):" +
            s"($loadedClassName, $version).  Supported:\n" +
            s"  ($classNameV1_0, 1.0)")
      }
    }
  }
